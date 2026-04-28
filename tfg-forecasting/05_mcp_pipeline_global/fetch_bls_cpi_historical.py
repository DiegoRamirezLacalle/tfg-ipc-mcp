"""
fetch_bls_cpi_historical.py
----------------------------
Descarga el histórico de publicaciones de CPI del BLS (US BLS CPI) 2002-2024.

Estrategia:
  1. Scrape https://www.bls.gov/schedule/news_release/cpi.htm
     para obtener fechas y URLs de publicaciones pasadas.
  2. Para cada mes, descargar la news release:
     https://www.bls.gov/news.release/archives/cpi_{MMDDYYYY}.htm
  3. Fallback: si el archivo no existe, intentar la URL actual de la release.

Nota: el BLS publica el CPI del mes M a mediados del mes M+1.
El shift +1 de leakage se aplica en news_to_features_global.py.

Salida: data/raw/global_raw/bls_cpi/{YYYY-MM}.json  (fecha del mes del CPI, no publicación)
Resumible: si YYYY-MM.json existe, se salta.
Rate limit: 1.5 s entre peticiones.

Uso:
  python fetch_bls_cpi_historical.py
  python fetch_bls_cpi_historical.py --start 2020
  python fetch_bls_cpi_historical.py --mongo
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
ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "global_raw" / "bls_cpi"

MONGO_URI  = "mongodb://localhost:27017"
MONGO_DB   = "tfg_ipc_mcp"
MONGO_COLL = "news_raw_global"

START_YEAR = 2002
END_YEAR   = 2024
RATE_LIMIT = 1.5

BLS_BASE     = "https://www.bls.gov"
BLS_SCHED    = BLS_BASE + "/schedule/news_release/cpi.htm"
BLS_ARCHIVE  = BLS_BASE + "/news.release/archives/cpi_{date}.htm"  # MMDDYYYY
BLS_CURRENT  = BLS_BASE + "/news.release/cpi.htm"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TFG-Academic-Research/1.0)"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Patrón fecha BLS en URLs: cpi_MMDDYYYY.htm
BLS_DATE_RE = re.compile(r"cpi_(\d{8})\.htm", re.IGNORECASE)


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


def _make_doc(cpi_month: str, pub_date: str, title: str, body: str, url: str) -> dict:
    """
    cpi_month: YYYY-MM del mes de CPI que se publica (p.ej. 2024-01 para el release de feb 2024)
    pub_date:  fecha exacta de publicación (YYYY-MM-DD)
    """
    return {
        "date":     f"{cpi_month}-01",  # primer día del mes del CPI
        "pub_date": pub_date,
        "title":    title.strip()[:300],
        "body":     body.strip()[:4000],
        "source":   "bls_cpi",
        "url":      url,
        "raw_source": "bls_news_release",
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
        return 0


def _fetch_html(url: str, client: httpx.Client) -> str:
    try:
        r = client.get(url)
        return r.text if r.status_code == 200 else ""
    except Exception as e:
        print(f"  [ERR] {url}: {e}")
        return ""


def _extract_release_text(html: str) -> str:
    """Extrae el texto principal de una news release del BLS."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    # El BLS tiene el contenido en <div id="content"> o <main>
    main = (
        soup.find("div", {"id": "content"})
        or soup.find("main")
        or soup.find("div", class_="highlight")
        or soup.find("article")
    )
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Solo primeras ~3000 chars (summary table + highlights)
    return text[:4000]


def _scrape_schedule_page(client: httpx.Client) -> list[tuple[str, str, str]]:
    """
    Scrape the BLS CPI schedule page to extract past release dates and URLs.
    Returns list of (pub_date YYYY-MM-DD, cpi_month YYYY-MM, url).
    """
    html = _fetch_html(BLS_SCHED, client)
    _sleep()
    if not html:
        print("  [WARN] No se pudo cargar el calendario BLS CPI")
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Buscar todos los links con patrón /news.release/archives/cpi_{MMDDYYYY}.htm
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = BLS_DATE_RE.search(href)
        if not m:
            continue

        raw_date = m.group(1)  # MMDDYYYY
        try:
            pub_dt = datetime.strptime(raw_date, "%m%d%Y")
        except ValueError:
            continue

        full_url = href if href.startswith("http") else BLS_BASE + href

        # El BLS publica CPI del mes anterior
        # Publicado en febrero → CPI de enero
        if pub_dt.month == 1:
            cpi_year = pub_dt.year - 1
            cpi_month = 12
        else:
            cpi_year = pub_dt.year
            cpi_month = pub_dt.month - 1

        cpi_ym = f"{cpi_year:04d}-{cpi_month:02d}"
        pub_date = pub_dt.strftime("%Y-%m-%d")

        results.append((pub_date, cpi_ym, full_url))

    return results


def _build_archive_url(year: int, month: int) -> list[str]:
    """
    Genera posibles URLs del archivo BLS para el CPI del mes (year, month).
    El CPI del mes M se publica a mediados del mes M+1.
    Intenta varios días típicos de publicación (12-17 de cada mes).
    """
    if month == 12:
        pub_year, pub_month = year + 1, 1
    else:
        pub_year, pub_month = year, month + 1

    # BLS típicamente publica entre los días 10 y 17 del mes siguiente
    urls = []
    for day in range(10, 18):
        try:
            date_str = f"{pub_month:02d}{day:02d}{pub_year:04d}"
            urls.append(BLS_ARCHIVE.format(date=date_str))
        except Exception:
            pass
    return urls


# ── Pipeline principal ───────────────────────────────────────────────────

def fetch_bls_cpi(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    use_mongo: bool = False,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"BLS CPI News Releases ({start_year}-{end_year})")
    print(f"{'='*60}")

    client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)
    all_new_docs: list[dict] = []
    total_new = total_skipped = total_errors = 0

    # 1. Scrape schedule page for known release dates
    print("\n[1/2] Scraping calendario de releases BLS...")
    schedule = _scrape_schedule_page(client)

    # Filtrar rango
    schedule_filtered = [
        (pd, ym, url) for pd, ym, url in schedule
        if start_year <= int(ym[:4]) <= end_year
    ]
    print(f"  Releases encontradas en calendario: {len(schedule_filtered)}")

    # Construir mapa cpi_month -> (pub_date, url)
    month_map: dict[str, tuple[str, str]] = {}
    for pub_date, cpi_ym, url in schedule_filtered:
        if cpi_ym not in month_map:
            month_map[cpi_ym] = (pub_date, url)

    # 2. Iterar meses 2002-01 a 2024-12
    print("\n[2/2] Descargando news releases...")
    import pandas as pd
    months = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq="MS",
    )

    for dt in months:
        ym = dt.strftime("%Y-%m")
        year = dt.year
        month = dt.month

        if _already_fetched(ym):
            total_skipped += 1
            continue

        # Intentar URL del calendario
        if ym in month_map:
            pub_date, url = month_map[ym]
            html = _fetch_html(url, client)
            _sleep()
        else:
            html = ""
            pub_date = ""
            url = ""

        # Fallback: intentar URLs candidatas
        if not html:
            candidate_urls = _build_archive_url(year, month)
            for candidate in candidate_urls:
                html = _fetch_html(candidate, client)
                _sleep()
                if html and len(html) > 500:
                    url = candidate
                    # Extraer fecha de publicación del URL
                    m = BLS_DATE_RE.search(candidate)
                    if m:
                        try:
                            pub_dt = datetime.strptime(m.group(1), "%m%d%Y")
                            pub_date = pub_dt.strftime("%Y-%m-%d")
                        except ValueError:
                            pub_date = ""
                    break
                html = ""

        if not html:
            total_errors += 1
            print(f"  {ym}: no disponible")
            # Crear placeholder
            doc = _make_doc(
                cpi_month=ym,
                pub_date=pub_date or f"{year if month < 12 else year+1}-{(month%12)+1:02d}-15",
                title=f"US CPI {ym}",
                body=f"CPI data for {ym}. Release not available via automated scraping.",
                url=url or BLS_BASE + "/news.release/cpi.htm",
            )
            _save(ym, [doc])
            all_new_docs.append(doc)
            continue

        body = _extract_release_text(html)
        if not body:
            body = f"CPI data for {ym}. Full text extraction failed."

        doc = _make_doc(
            cpi_month=ym,
            pub_date=pub_date,
            title=f"US BLS CPI News Release — {ym}",
            body=body,
            url=url,
        )
        _save(ym, [doc])
        all_new_docs.append(doc)
        total_new += 1
        print(f"  {ym}: OK  (pub: {pub_date}, {len(body):,} chars)")

    client.close()

    if use_mongo and all_new_docs:
        n = _insert_mongo(all_new_docs)
        print(f"\n  MongoDB: {n} docs insertados")

    print(f"\n{'='*60}")
    print(f"BLS CPI: {total_new} nuevos, {total_skipped} existentes, {total_errors} sin texto")
    print(f"Archivos en: {RAW_DIR}")
    print(f"{'='*60}")
    return all_new_docs


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga histórico de BLS CPI news releases (2002-2024)"
    )
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end",   type=int, default=END_YEAR)
    parser.add_argument("--mongo", action="store_true")
    args = parser.parse_args()
    fetch_bls_cpi(start_year=args.start, end_year=args.end, use_mongo=args.mongo)


if __name__ == "__main__":
    main()
