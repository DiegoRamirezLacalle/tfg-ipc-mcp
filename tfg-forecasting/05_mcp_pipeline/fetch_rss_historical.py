"""Download historical official press releases (2015-2024) from three sources:

  BCE — ECB monetary policy decisions from press releases.
  INE — Monthly CPI press releases via predictable URL.
  BdE — Banco de Espana press releases via RSS.

Each release is saved as normalized JSON in:
  data/raw/rss_raw/{source}/YYYY-MM.json

Resumable: skips months where YYYY-MM.json already exists.
Rate limiting: 1 second between HTTP requests.

Usage:
  python fetch_rss_historical.py                     # all sources
  python fetch_rss_historical.py --source bce        # BCE only
  python fetch_rss_historical.py --source ine        # INE only
  python fetch_rss_historical.py --source bde        # BdE only
  python fetch_rss_historical.py --mongo             # also insert into MongoDB

Requirements:
  pip install httpx beautifulsoup4 feedparser pymongo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx
from bs4 import BeautifulSoup
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BASE = PROJECT_ROOT / "data" / "raw" / "rss_raw"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "tfg_ipc_mcp"
MONGO_COLLECTION = "news_raw"

START_YEAR = 2015
END_YEAR = 2024

RATE_LIMIT = 1.0  # seconds between requests

# headers to avoid server blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TFG-IPC-MCP-Research/1.0; "
        "+academic-research)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def _rate_limit():
    """Sleep between requests to avoid overloading servers."""
    time.sleep(RATE_LIMIT)


def _save_json(source: str, year_month: str, docs: list[dict]):
    """Save document list as JSON to data/raw/rss_raw/{source}/YYYY-MM.json."""
    out_dir = RAW_BASE / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year_month}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    return path


def _already_fetched(source: str, year_month: str) -> bool:
    """Return True if JSON for the given month already exists."""
    path = RAW_BASE / source / f"{year_month}.json"
    return path.exists()


def _make_doc(date: str, title: str, body: str, source: str, url: str) -> dict:
    """Create a normalized document dict."""
    return {
        "date": date,
        "title": title.strip(),
        "body": body.strip(),
        "source": source,
        "url": url,
        "raw_source": "rss_historical",
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "processed": False,
    }


def _insert_mongo(docs: list[dict]) -> int:
    """Insert documents into MongoDB, skipping duplicates by URL."""
    if not docs:
        return 0
    client = MongoClient(MONGO_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]
    col.create_index("url", unique=True, sparse=True)

    inserted = 0
    for doc in docs:
        try:
            col.insert_one(doc)
            inserted += 1
        except Exception:
            pass  # duplicate
    return inserted


# BCE: ECB monetary policy decisions

# ECB Governing Council monetary policy meeting dates (2015-2024).
# Source: official ECB calendars published annually.
# Includes only meetings with rate decisions (not "non-monetary" meetings).
BCE_MEETING_DATES = [
    # 2015
    "2015-01-22", "2015-03-05", "2015-04-15", "2015-06-03",
    "2015-07-16", "2015-09-03", "2015-10-22", "2015-12-03",
    # 2016
    "2016-01-21", "2016-03-10", "2016-04-21", "2016-06-02",
    "2016-07-21", "2016-09-08", "2016-10-20", "2016-12-08",
    # 2017
    "2017-01-19", "2017-03-09", "2017-04-27", "2017-06-08",
    "2017-07-20", "2017-09-07", "2017-10-26", "2017-12-14",
    # 2018
    "2018-01-25", "2018-03-08", "2018-04-26", "2018-06-14",
    "2018-07-26", "2018-09-13", "2018-10-25", "2018-12-13",
    # 2019
    "2019-01-24", "2019-03-07", "2019-04-10", "2019-06-06",
    "2019-07-25", "2019-09-12", "2019-10-24", "2019-12-12",
    # 2020
    "2020-01-23", "2020-03-12", "2020-04-30", "2020-06-04",
    "2020-07-16", "2020-09-10", "2020-10-29", "2020-12-10",
    # 2021
    "2021-01-21", "2021-03-11", "2021-04-22", "2021-06-10",
    "2021-07-22", "2021-09-09", "2021-10-28", "2021-12-16",
    # 2022
    "2022-02-03", "2022-03-10", "2022-04-14", "2022-06-09",
    "2022-07-21", "2022-09-08", "2022-10-27", "2022-12-15",
    # 2023
    "2023-02-02", "2023-03-16", "2023-05-04", "2023-06-15",
    "2023-07-27", "2023-09-14", "2023-10-26", "2023-12-14",
    # 2024
    "2024-01-25", "2024-03-07", "2024-04-11", "2024-06-06",
    "2024-07-18", "2024-09-12", "2024-10-17", "2024-12-12",
]

BCE_BASE_URL = "https://www.ecb.europa.eu"
BCE_PR_INDEX = BCE_BASE_URL + "/press/pr/date/{year}/html/index.en.html"
BCE_RSS_URL = "https://www.ecb.europa.eu/rss/press.html"


def _fetch_bce_from_rss() -> list[dict]:
    """Fetch recent ECB press releases via RSS."""
    feed = feedparser.parse(BCE_RSS_URL)
    docs = []
    for entry in feed.entries:
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        summary = getattr(entry, "summary", "")
        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        is_mopo = any(kw in title.lower() for kw in [
            "monetary policy", "interest rate", "key ecb interest",
            "governing council", "types", "tipos",
        ])
        if not is_mopo and "mp" not in link:
            continue

        docs.append(_make_doc(
            date=pub_date.strftime("%Y-%m-%d") if pub_date else "",
            title=title,
            body=summary,
            source="bce",
            url=link,
        ))
    return docs


def _fetch_bce_press_release(url: str, client: httpx.Client) -> str:
    """Extract full text from an ECB press release page."""
    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="section")
            or soup.find("div", {"id": "main-wrapper"})
        )
        if content:
            return content.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)[:3000]
    except Exception:
        return ""


def _discover_bce_mopo_links(year: int, client: httpx.Client) -> list[dict]:
    """Discover monetary policy decision links from the ECB annual press release index."""
    url = BCE_PR_INDEX.format(year=year)
    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if "ecb.mp" in href or "monetary policy" in text.lower():
                full_url = href if href.startswith("http") else BCE_BASE_URL + href
                links.append({"url": full_url, "title": text})

        return links
    except Exception:
        return []


def fetch_bce(
    use_mongo: bool = False,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
):
    """Download historical ECB monetary policy decisions.

    Strategy:
    1. Attempt to discover links from the annual press release index.
    2. If unavailable (JS-rendered), fall back to known Governing Council
       meeting dates + RSS feed.
    """
    logger.info(f"BCE: downloading monetary policy decisions ({start_year}-{end_year})")

    all_docs: list[dict] = []
    total_new = 0
    client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)

    logger.info("[1/2] Searching annual press release index...")
    discovered_links = []
    for year in range(start_year, end_year + 1):
        links = _discover_bce_mopo_links(year, client)
        if links:
            logger.info(f"  {year}: {len(links)} monetary policy links")
            discovered_links.extend([(year, l) for l in links])
        else:
            logger.info(f"  {year}: index not available (JS-rendered)")
        _rate_limit()

    logger.info("[2/2] Processing known Governing Council dates...")
    filtered_dates = [
        d for d in BCE_MEETING_DATES
        if start_year <= int(d[:4]) <= end_year
    ]
    for date_str in filtered_dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        ym = dt.strftime("%Y-%m")

        if _already_fetched("bce", ym):
            continue

        month_docs = []
        for year, link_info in discovered_links:
            if str(year) == str(dt.year):
                body = _fetch_bce_press_release(link_info["url"], client)
                if body:
                    month_docs.append(_make_doc(
                        date=date_str,
                        title=link_info["title"],
                        body=body[:3000],
                        source="bce",
                        url=link_info["url"],
                    ))
                    _rate_limit()

        if not month_docs:
            month_docs.append(_make_doc(
                date=date_str,
                title=f"ECB Governing Council monetary policy decision - {date_str}",
                body=f"Monetary policy meeting held on {date_str}. "
                     f"Full text not available via automated scraping.",
                source="bce",
                url=f"https://www.ecb.europa.eu/press/govcdec/mopo/{dt.year}/",
            ))

        _save_json("bce", ym, month_docs)
        all_docs.extend(month_docs)
        total_new += 1
        scraped = month_docs[0]["body"] and "not available" not in month_docs[0]["body"]
        logger.info(f"  {ym}: {len(month_docs)} docs ({'scraped' if scraped else 'placeholder'})")

    logger.info("  Complementing with recent RSS...")
    rss_docs = _fetch_bce_from_rss()
    for doc in rss_docs:
        if doc["date"]:
            ym = doc["date"][:7]
            if not _already_fetched("bce", ym):
                _save_json("bce", ym, [doc])
                all_docs.append(doc)
                total_new += 1
                logger.info(f"  {ym}: RSS - {doc['title'][:50]}...")

    client.close()

    if use_mongo:
        n = _insert_mongo(all_docs)
        logger.info(f"  MongoDB: {n} docs inserted")

    logger.info(f"  BCE total: {total_new} new months")
    return all_docs


# INE: CPI press releases

# Predictable URL: https://www.ine.es/dyngs/Prensa/IPCMMYY.htm
INE_BASE_URL = "https://www.ine.es/dyngs/Prensa/IPC{mm}{yy}.htm"


def _fetch_ine_month(year: int, month: int, client: httpx.Client) -> dict | None:
    """Download and parse one INE CPI press release."""
    yy = f"{year % 100:02d}"
    mm = f"{month:02d}"
    url = INE_BASE_URL.format(mm=mm, yy=yy)

    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else f"IPC {month:02d}/{year}"

        content = (
            soup.find("div", class_="contenido")
            or soup.find("div", {"id": "contenido"})
            or soup.find("main")
            or soup.find("article")
        )
        if content:
            body = content.get_text(separator="\n", strip=True)
        else:
            for nav in soup.find_all(["nav", "header", "footer", "script", "style"]):
                nav.decompose()
            body = soup.get_text(separator="\n", strip=True)

        body = body[:5000]
        date_str = f"{year}-{month:02d}-01"
        return _make_doc(date=date_str, title=title, body=body, source="ine", url=url)
    except Exception as e:
        logger.warning(f"{url}: {e}")
        return None


def fetch_ine(
    use_mongo: bool = False,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
):
    """Download historical INE CPI press releases.

    URL pattern: https://www.ine.es/dyngs/Prensa/IPCMMYY.htm
    """
    logger.info(f"INE: downloading CPI press releases ({start_year}-{end_year})")

    all_docs: list[dict] = []
    total_new = 0
    total_skipped = 0
    total_errors = 0
    client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            ym = f"{year}-{month:02d}"

            if _already_fetched("ine", ym):
                total_skipped += 1
                continue

            doc = _fetch_ine_month(year, month, client)
            _rate_limit()

            if doc:
                _save_json("ine", ym, [doc])
                all_docs.append(doc)
                total_new += 1
                body_preview = doc["body"][:60].replace("\n", " ")
                logger.info(f"  {ym}: OK - {body_preview}...")
            else:
                total_errors += 1
                logger.info(f"  {ym}: not available")

    client.close()

    if use_mongo:
        n = _insert_mongo(all_docs)
        logger.info(f"  MongoDB: {n} docs inserted")

    logger.info(f"  INE total: {total_new} new, {total_skipped} existing, {total_errors} errors")
    return all_docs


# INE PDFs: CPI press releases from PDFs (2015-2024)

# Confirmed URL: https://www.ine.es/daco/daco42/daco421/ipc{MM}{YY}.pdf
# Examples: ipc0115.pdf (Jan-2015), ipc1224.pdf (Dec-2024)
INE_PDF_BASE = "https://www.ine.es/daco/daco42/daco421/ipc{mm}{yy}.pdf"


def _extract_ine_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from INE CPI press release PDFs."""
    import io
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[:3]:  # first 3 pages are sufficient
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def _normalize_ine_pdf_text(text: str) -> str:
    """Normalize PDF text: fix mojibake, unify Unicode minus signs, standardize decimal separator."""
    repaired = text

    # Fix common mojibake in PDFs (e.g. "variaciÃ³n", "â€"0,4").
    if ("Ã" in repaired) or ("â" in repaired):
        try:
            candidate = repaired.encode("latin1", errors="ignore").decode(
                "utf-8", errors="ignore"
            )
            if candidate and candidate.count("�") <= repaired.count("�"):
                repaired = candidate
        except Exception:
            pass

    repaired = unicodedata.normalize("NFKC", repaired)

    # Unify minus/dash to ASCII '-' so regex captures negatives.
    for ch in ["−", "–", "—", "‑", "‒", "―", "﹣"]:
        repaired = repaired.replace(ch, "-")
    for bad in ["â€"", "â€"", "âˆ'"]:
        repaired = repaired.replace(bad, "-")

    # Non-breaking spaces and tabs.
    repaired = repaired.replace(" ", " ").replace(" ", " ").replace("\t", " ")

    # Normalize Spanish decimal separator.
    repaired = repaired.replace(",", ".")

    return repaired


def _parse_ine_pdf_signals(text: str) -> dict:
    """Extract CPI signals from PDF text using regex (no LLM — data is numeric)."""
    text_norm = _normalize_ine_pdf_text(text)
    number = r"([+\-]?\s*\d+(?:\.\d+)?)"

    # Annual rate (CPI general)
    annual_rate = None
    for pattern in [
        rf"(?:variaci\w*|tasa)\s+anual[^\d+\-]{{0,120}}{number}\s*(?:por\s*ciento|%)",
        rf"tasa\s+de\s+variaci\w*\s+anual[^\d+\-]{{0,120}}{number}\s*(?:por\s*ciento|%)",
        rf"ipc[^\n]{{0,120}}anual[^\d+\-]{{0,120}}{number}\s*(?:por\s*ciento|%)",
    ]:
        m = re.search(pattern, text_norm, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(" ", ""))
                if -5.0 <= val <= 15.0:  # reasonable range for Spanish CPI
                    annual_rate = val
                    break
            except ValueError:
                continue

    # Monthly rate
    monthly_rate = None
    for pattern in [
        rf"(?:variaci\w*|tasa)\s+mensual[^\d+\-]{{0,120}}{number}\s*(?:por\s*ciento|%)",
        rf"tasa\s+de\s+variaci\w*\s+mensual[^\d+\-]{{0,120}}{number}\s*(?:por\s*ciento|%)",
        rf"mensual[^\d+\-]{{0,120}}{number}\s*(?:por\s*ciento|%)",
    ]:
        m = re.search(pattern, text_norm, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(" ", ""))
                if -3.0 <= val <= 3.0:  # reasonable monthly range
                    monthly_rate = val
                    break
            except ValueError:
                continue

    # Build signals
    if annual_rate is None and monthly_rate is None:
        # Could not extract rates — return neutral defaults
        return {
            "decision": "dato",
            "magnitude": None,
            "tone": "neutral",
            "shock_score": 0.0,
            "uncertainty_index": 0.5,
            "topic": "inflacion",
            "ipc_general": None,
            "ipc_monthly": None,
        }

    # Decision based on monthly change
    if monthly_rate is not None:
        if monthly_rate > 0.3:
            decision = "subida"
        elif monthly_rate < -0.3:
            decision = "bajada"
        else:
            decision = "dato"
        shock = round(min(abs(monthly_rate) / 2.0, 1.0), 2)
        magnitude = abs(monthly_rate)
    else:
        decision = "dato"
        shock = 0.0
        magnitude = None

    # Tone based on annual inflation level
    if annual_rate is not None:
        if annual_rate > 3.5:
            tone = "negativo"
        elif annual_rate < 1.5:
            tone = "positivo"
        else:
            tone = "neutral"
    else:
        tone = "neutral"

    return {
        "decision": decision,
        "magnitude": round(magnitude, 2) if magnitude is not None else None,
        "tone": tone,
        "shock_score": shock,
        "uncertainty_index": 0.3,  # CPI data is objective, low uncertainty
        "topic": "inflacion",
        "ipc_general": annual_rate,
        "ipc_monthly": monthly_rate,
    }


def fetch_ine_pdfs(
    use_mongo: bool = True,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
):
    """Download 120 INE CPI press release PDFs (2015-2024) and extract rates with regex.

    URL: https://www.ine.es/daco/daco42/daco421/ipc{MM}{YY}.pdf
    Resumable: skips months already in MongoDB.
    """
    import pdfplumber as _pdf_check  # verify availability at startup
    del _pdf_check

    logger.info(f"INE PDFs: downloading CPI press releases ({start_year}-{end_year})")

    client_http = httpx.Client(headers=HEADERS, timeout=60.0, follow_redirects=True)
    mongo_client = MongoClient(MONGO_URI)
    col = mongo_client[MONGO_DB][MONGO_COLLECTION]
    col.create_index("url", unique=True, sparse=True)

    total_new = 0
    total_skipped = 0
    total_errors = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            ym = f"{year}-{month:02d}"
            yy = f"{year % 100:02d}"
            mm = f"{month:02d}"
            url = INE_PDF_BASE.format(mm=mm, yy=yy)

            if col.count_documents({"url": url}) > 0:
                total_skipped += 1
                continue

            try:
                resp = client_http.get(url)
                _rate_limit()

                if resp.status_code != 200:
                    logger.info(f"  {ym}: HTTP {resp.status_code} — not available")
                    total_errors += 1
                    continue

                if len(resp.content) < 1000:
                    logger.info(f"  {ym}: response too small ({len(resp.content)}b)")
                    total_errors += 1
                    continue

                text = _extract_ine_pdf_text(resp.content)
                if not text or len(text) < 50:
                    logger.info(f"  {ym}: PDF has no extractable text")
                    total_errors += 1
                    continue

                signals = _parse_ine_pdf_signals(text)
                ipc_str = (
                    f"CPI annual={signals['ipc_general']}%"
                    if signals["ipc_general"] is not None
                    else "rate not parsed"
                )

                doc = {
                    "date": f"{year}-{month:02d}-01",
                    "title": f"INE IPC nota de prensa {ym}",
                    "body": text[:3000],
                    "source": "ine",
                    "url": url,
                    "raw_source": "pdf_historical",
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                    "processed": True,
                    "signals": signals,
                }

                if use_mongo:
                    try:
                        col.insert_one(doc)
                    except Exception:
                        pass  # duplicate

                doc_json = {k: v for k, v in doc.items() if k != "_id"}
                _save_json("ine_pdf", ym, [doc_json])

                total_new += 1
                logger.info(
                    f"  {ym}: OK — {ipc_str} | {signals['tone']} | shock={signals['shock_score']}"
                )

            except Exception as e:
                logger.warning(f"  {ym}: ERROR — {e}")
                total_errors += 1
                _rate_limit()

    client_http.close()
    mongo_client.close()

    logger.info(f"  INE PDFs: {total_new} new, {total_skipped} existing, {total_errors} errors")
    return total_new


# BdE: Banco de Espana press releases

BDE_RSS_URL = "https://www.bde.es/rss/es/"

# Keywords to filter relevant notes (Spanish text matching — must stay in Spanish)
BDE_KEYWORDS = [
    "tipo", "interes", "inflacion", "ipc", "precio",
    "politica monetaria", "euribor", "credito", "hipoteca",
    "estabilidad", "financiera", "supervision", "bancaria",
]


def fetch_bde(use_mongo: bool = False):
    """Download Banco de Espana press releases via RSS.

    Known limitation: BdE RSS only retains ~20-50 articles. Full history
    would require Selenium/Playwright (BdE site uses heavy JS rendering).
    """
    logger.info("BdE: downloading Banco de Espana press releases (RSS)")

    all_docs: list[dict] = []

    rss_urls = [
        BDE_RSS_URL,
        "https://www.bde.es/rss/es/notas-prensa.xml",
        "https://www.bde.es/rss/es/todas-las-noticias.xml",
    ]

    entries = []
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            if feed.entries:
                entries.extend(feed.entries)
                logger.info(f"  RSS {rss_url}: {len(feed.entries)} entries")
        except Exception as e:
            logger.warning(f"  RSS {rss_url}: error - {e}")

    if not entries:
        logger.warning("No BdE RSS entries obtained.")
        logger.warning(
            "Known limitation: BdE site uses JS rendering. "
            "Full history requires Selenium/Playwright."
        )
        return all_docs

    seen_months: set[str] = set()
    for entry in entries:
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "")

        text_lower = (title + " " + summary).lower()
        is_relevant = any(kw in text_lower for kw in BDE_KEYWORDS)
        if not is_relevant:
            continue

        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        if not pub_date:
            continue

        ym = pub_date.strftime("%Y-%m")
        doc = _make_doc(
            date=pub_date.strftime("%Y-%m-%d"),
            title=title,
            body=summary,
            source="bde",
            url=link,
        )

        if ym not in seen_months:
            seen_months.add(ym)
            if not _already_fetched("bde", ym):
                _save_json("bde", ym, [doc])
                all_docs.append(doc)
                logger.info(f"  {ym}: {title[:60]}...")

    if use_mongo:
        n = _insert_mongo(all_docs)
        logger.info(f"  MongoDB: {n} docs inserted")

    logger.info(f"  BdE total: {len(all_docs)} new months (limited by RSS)")
    return all_docs


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Download historical official press releases (2015-2024)"
    )
    parser.add_argument(
        "--source",
        choices=["bce", "ine", "ine-pdf", "bde"],
        default=None,
        help="Download one source only (default: all). 'ine-pdf' downloads historical PDFs.",
    )
    parser.add_argument(
        "--mongo",
        action="store_true",
        help="Also insert documents into MongoDB",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"Start year (default: {START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"End year (default: {END_YEAR})",
    )

    args = parser.parse_args()

    start_y = args.start_year
    end_y = args.end_year
    sources = [args.source] if args.source else ["bce", "ine-pdf", "bde"]

    logger.info(f"Historical press release download: {start_y}-{end_y}")
    logger.info(f"Sources: {', '.join(sources)}, rate limit: {RATE_LIMIT}s")
    logger.info(f"Output: {RAW_BASE}/")

    if "bce" in sources:
        fetch_bce(use_mongo=args.mongo, start_year=start_y, end_year=end_y)

    if "ine" in sources:
        fetch_ine(use_mongo=args.mongo, start_year=start_y, end_year=end_y)

    if "ine-pdf" in sources:
        fetch_ine_pdfs(use_mongo=True, start_year=start_y, end_year=end_y)

    if "bde" in sources:
        fetch_bde(use_mongo=args.mongo)

    logger.info(f"Download complete. Files in: {RAW_BASE}")
    if args.mongo:
        logger.info(f"Documents inserted in MongoDB: {MONGO_DB}.{MONGO_COLLECTION}")


if __name__ == "__main__":
    main()
