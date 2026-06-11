"""
fetch_fomc_historical.py
------------------------
Descarga el histórico de comunicados del FOMC (Federal Reserve) 2002-2024.

Estrategia:
  1. Scrape https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
     (incluye últimos ~3 años con links directos a statements).
  2. Scrape páginas anuales históricas:
     https://www.federalreserve.gov/monetarypolicy/fomc{YYYY}.htm
  3. Para cada reunión, descargar el statement:
     https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm

Salida: data/raw/global_raw/fomc/{YYYY-MM}.json (un fichero por mes)
Resumible: si YYYY-MM.json ya existe, se salta.
Rate limit: 1.5 s entre peticiones.

Uso:
  python fetch_fomc_historical.py                     # 2002-2024
  python fetch_fomc_historical.py --start 2020        # desde 2020
  python fetch_fomc_historical.py --mongo             # también inserta en MongoDB
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "global_raw" / "fomc"

MONGO_URI  = "mongodb://localhost:27017"
MONGO_DB   = "tfg_ipc_mcp"
MONGO_COLL = "news_raw_global"

START_YEAR = 2002
END_YEAR   = 2024
RATE_LIMIT = 1.5  # segundos

FED_BASE  = "https://www.federalreserve.gov"
CALENDAR  = FED_BASE + "/monetarypolicy/fomccalendars.htm"
HIST_INDEX = FED_BASE + "/monetarypolicy/fomc_historical.htm"
YEAR_PAGE  = FED_BASE + "/monetarypolicy/fomc{year}.htm"
STMT_URL   = FED_BASE + "/newsevents/pressreleases/monetary{date}a.htm"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TFG-Academic-Research/1.0; "
        "contact: research@example.com)"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DATE_RE = re.compile(r"(\d{8})")

# Fechas FOMC conocidas 2002-2020 (fallback cuando el scraping falla)
# Las URLs siguen el patrón: monetary{YYYYMMDD}a.htm
FOMC_KNOWN_DATES = [
    # 2002
    "20020130", "20020319", "20020507", "20020626", "20020813",
    "20020924", "20021106", "20021210",
    # 2003
    "20030129", "20030318", "20030506", "20030625", "20030812",
    "20030916", "20031028", "20031209",
    # 2004
    "20040128", "20040316", "20040504", "20040630", "20040810",
    "20040921", "20041110", "20041214",
    # 2005
    "20050202", "20050322", "20050503", "20050630", "20050809",
    "20050920", "20051101", "20051213",
    # 2006
    "20060131", "20060328", "20060510", "20060629", "20060808",
    "20060920", "20061025", "20061212",
    # 2007
    "20070131", "20070321", "20070509", "20070628", "20070807",
    "20070918", "20071031", "20071211",
    # 2008
    "20080122", "20080130", "20080318", "20080430", "20080625",
    "20080805", "20080916", "20081008", "20081029", "20081216",
    # 2009
    "20090128", "20090318", "20090429", "20090624", "20090812",
    "20090923", "20091104", "20091216",
    # 2010
    "20100127", "20100316", "20100428", "20100623", "20100810",
    "20100921", "20101103", "20101214",
    # 2011
    "20110126", "20110315", "20110427", "20110622", "20110809",
    "20110921", "20111102", "20111213",
    # 2012
    "20120125", "20120313", "20120425", "20120620", "20120801",
    "20120913", "20121024", "20121212",
    # 2013
    "20130130", "20130320", "20130501", "20130619", "20130731",
    "20130918", "20131030", "20131218",
    # 2014
    "20140129", "20140319", "20140430", "20140618", "20140730",
    "20140917", "20141029", "20141217",
    # 2015
    "20150128", "20150318", "20150429", "20150617", "20150729",
    "20150917", "20151028", "20151216",
    # 2016
    "20160127", "20160316", "20160427", "20160615", "20160727",
    "20160921", "20161102", "20161214",
    # 2017
    "20170201", "20170315", "20170503", "20170614", "20170726",
    "20170920", "20171101", "20171213",
    # 2018
    "20180131", "20180321", "20180502", "20180613", "20180801",
    "20180926", "20181108", "20181219",
    # 2019
    "20190130", "20190320", "20190501", "20190619", "20190731",
    "20190918", "20191030", "20191211",
    # 2020
    "20200129", "20200303", "20200315", "20200429", "20200610",
    "20200729", "20200916", "20201105", "20201216",
]


# ── Utilidades ──────────────────────────────────────────────────────────

def _sleep():
    time.sleep(RATE_LIMIT)


def _already_fetched(ym: str) -> bool:
    return (RAW_DIR / f"{ym}.json").exists()


def _save(ym: str, docs: list[dict]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ym}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    return path


def _make_doc(date: str, title: str, body: str, url: str) -> dict:
    return {
        "date":  date,
        "title": title.strip()[:300],
        "body":  body.strip()[:4000],
        "source": "fomc",
        "url":   url,
        "raw_source": "fomc_statement",
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "processed": False,
    }


def _insert_mongo(docs: list[dict]):
    try:
        from pymongo import MongoClient
        col = MongoClient(MONGO_URI)[MONGO_DB][MONGO_COLL]
        col.create_index("url", unique=True, sparse=True)
        n = 0
        for doc in docs:
            try:
                col.insert_one({k: v for k, v in doc.items() if k != "_id"})
                n += 1
            except Exception:
                pass
        return n
    except ImportError:
        print("  [WARN] pymongo no instalado - omitiendo MongoDB")
        return 0


# ── Scraping ────────────────────────────────────────────────────────────

def _fetch_html(url: str, client: httpx.Client) -> str:
    try:
        r = client.get(url)
        return r.text if r.status_code == 200 else ""
    except Exception as e:
        print(f"  [ERR] {url}: {e}")
        return ""


def _extract_statement_text(html: str) -> str:
    """Extrae el texto limpio de un FOMC statement HTML."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # Eliminar elementos de navegación
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    main = (
        soup.find("div", {"id": "article"})
        or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="row")
    )
    if main:
        return main.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def _find_statement_links_in_page(html: str) -> list[str]:
    """Extrae links a FOMC statements (patrón /monetary{YYYYMMDD}a.htm)."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Patrón: monetary{YYYYMMDD}a.htm
        if re.search(r"monetary\d{8}a\.htm", href):
            full = href if href.startswith("http") else FED_BASE + href
            links.append(full)
    return list(dict.fromkeys(links))  # deduplicar manteniendo orden


def _discover_all_statement_links(
    client: httpx.Client,
    start_year: int,
    end_year: int,
) -> list[str]:
    """
    Recopila links a todos los statements del FOMC para el rango de años.
    Combina el calendario actual + páginas anuales históricas.
    """
    all_links: list[str] = []

    print("  Scraping calendario actual (fomccalendars.htm)...")
    html = _fetch_html(CALENDAR, client)
    _sleep()
    all_links += _find_statement_links_in_page(html)

    print("  Scraping páginas anuales históricas...")
    for year in range(start_year, end_year + 1):
        url = YEAR_PAGE.format(year=year)
        html = _fetch_html(url, client)
        _sleep()
        links = _find_statement_links_in_page(html)
        if links:
            print(f"    {year}: {len(links)} statements encontrados")
            all_links += links
        else:
            print(f"    {year}: sin links (intentando statement directo)")

    # Deduplicar
    return list(dict.fromkeys(all_links))


def _date_from_stmt_url(url: str) -> str | None:
    """Extrae YYYY-MM-DD del URL del statement."""
    m = DATE_RE.search(url)
    if not m:
        return None
    raw = m.group(1)  # YYYYMMDD
    try:
        dt = datetime.strptime(raw, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


# ── Pipeline principal ───────────────────────────────────────────────────

def fetch_fomc(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    use_mongo: bool = False,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"FOMC Statements - Federal Reserve ({start_year}-{end_year})")
    print(f"{'='*60}")

    client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)
    all_new_docs: list[dict] = []
    total_new = 0
    total_skipped = 0
    total_errors = 0

    # 1. Descubrir links via scraping (calendario actual + páginas año)
    print("\n[1/3] Descubriendo links a statements via scraping...")
    stmt_links = _discover_all_statement_links(client, start_year, end_year)

    # 2. Completar con fechas conocidas (fallback para años pre-2021)
    print("\n[2/3] Añadiendo URLs directas de fechas conocidas (2002-2020)...")
    known_urls = []
    for raw_date in FOMC_KNOWN_DATES:
        year = int(raw_date[:4])
        if start_year <= year <= end_year:
            url = STMT_URL.format(date=raw_date)
            known_urls.append(url)

    all_candidate_urls = list(dict.fromkeys(stmt_links + known_urls))

    # Filtrar por rango de años
    filtered = []
    for url in all_candidate_urls:
        m = DATE_RE.search(url)
        if m:
            year = int(m.group(1)[:4])
            if start_year <= year <= end_year:
                filtered.append(url)

    print(f"  Total candidatos: {len(filtered)} URLs")

    # 3. Descargar cada statement
    print("\n[3/3] Descargando statements...")
    seen_months: set[str] = set()

    for url in sorted(filtered):
        date_str = _date_from_stmt_url(url)
        if not date_str:
            continue

        ym = date_str[:7]  # YYYY-MM

        if ym in seen_months:
            continue  # solo una reunión por mes (la primera del sorted)
        seen_months.add(ym)

        if _already_fetched(ym):
            total_skipped += 1
            continue

        html = _fetch_html(url, client)
        _sleep()

        if not html:
            total_errors += 1
            print(f"  {ym}: HTTP error")
            continue

        body = _extract_statement_text(html)
        if not body or len(body) < 100:
            total_errors += 1
            print(f"  {ym}: texto vacío")
            continue

        doc = _make_doc(
            date=date_str,
            title=f"FOMC Statement {date_str}",
            body=body,
            url=url,
        )
        _save(ym, [doc])
        all_new_docs.append(doc)
        total_new += 1
        print(f"  {ym}: OK  ({len(body):,} chars)")

    client.close()

    if use_mongo and all_new_docs:
        n = _insert_mongo(all_new_docs)
        print(f"\n  MongoDB: {n} docs insertados")

    print(f"\n{'='*60}")
    print(f"FOMC: {total_new} nuevos, {total_skipped} existentes, {total_errors} errores")
    print(f"Archivos en: {RAW_DIR}")
    print(f"{'='*60}")
    return all_new_docs


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga histórico de FOMC statements (2002-2024)"
    )
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end",   type=int, default=END_YEAR)
    parser.add_argument("--mongo", action="store_true", help="Insertar en MongoDB")
    args = parser.parse_args()
    fetch_fomc(start_year=args.start, end_year=args.end, use_mongo=args.mongo)


if __name__ == "__main__":
    main()
