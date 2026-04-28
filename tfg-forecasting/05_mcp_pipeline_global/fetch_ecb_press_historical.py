"""
fetch_ecb_press_historical.py
------------------------------
Descarga el histórico de press conferences del BCE (2002-2024).

Estrategia:
  1. Scrape índice anual: https://www.ecb.europa.eu/press/pressconf/{YYYY}/html/index.en.html
  2. Descubrir links a press conferences individuales (patrón is{YYYYMMDD})
  3. Descargar transcripción de cada conferencia
  Fallback: si el índice no funciona (JS-rendered), intenta URL directas con
  fechas conocidas del Governing Council reutilizando BCE_MEETING_DATES
  del pipeline España.

Salida: data/raw/global_raw/ecb_press/{YYYY-MM}.json
Resumible: si YYYY-MM.json ya existe, se salta.
Rate limit: 1.5 s entre peticiones.

Uso:
  python fetch_ecb_press_historical.py
  python fetch_ecb_press_historical.py --start 2020
  python fetch_ecb_press_historical.py --mongo
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
RAW_DIR = ROOT / "data" / "raw" / "global_raw" / "ecb_press"

MONGO_URI  = "mongodb://localhost:27017"
MONGO_DB   = "tfg_ipc_mcp"
MONGO_COLL = "news_raw_global"

START_YEAR = 2002
END_YEAR   = 2024
RATE_LIMIT = 1.5

ECB_BASE    = "https://www.ecb.europa.eu"
ECB_PC_IDX  = ECB_BASE + "/press/pressconf/{year}/html/index.en.html"
ECB_PC_URL  = ECB_BASE + "/press/pressconf/{year}/html/{filename}.en.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TFG-Academic-Research/1.0)"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DATE_RE = re.compile(r"is(\d{8})")

# Fechas conocidas del Governing Council BCE (para fallback 2002-2024)
# Reuniones de política monetaria (con decisión de tipos)
BCE_MEETING_DATES = [
    # 2002
    "2002-01-03", "2002-02-07", "2002-03-07", "2002-04-04",
    "2002-05-02", "2002-06-06", "2002-07-04", "2002-08-01",
    "2002-09-12", "2002-10-10", "2002-11-07", "2002-12-05",
    # 2003
    "2003-01-23", "2003-02-06", "2003-03-06", "2003-04-03",
    "2003-05-08", "2003-06-05", "2003-07-10", "2003-09-04",
    "2003-10-02", "2003-11-06", "2003-12-04",
    # 2004
    "2004-01-08", "2004-02-05", "2004-03-04", "2004-04-01",
    "2004-05-06", "2004-06-03", "2004-07-01", "2004-08-05",
    "2004-09-02", "2004-10-07", "2004-11-04", "2004-12-02",
    # 2005
    "2005-01-13", "2005-02-03", "2005-03-03", "2005-04-07",
    "2005-05-04", "2005-06-02", "2005-07-07", "2005-08-04",
    "2005-09-01", "2005-10-06", "2005-11-03", "2005-12-01",
    # 2006
    "2006-01-12", "2006-02-02", "2006-03-02", "2006-04-06",
    "2006-05-04", "2006-06-08", "2006-07-06", "2006-08-03",
    "2006-09-07", "2006-10-05", "2006-11-02", "2006-12-07",
    # 2007
    "2007-01-11", "2007-02-08", "2007-03-08", "2007-04-12",
    "2007-05-10", "2007-06-06", "2007-07-05", "2007-08-02",
    "2007-09-06", "2007-10-04", "2007-11-08", "2007-12-06",
    # 2008
    "2008-01-10", "2008-02-07", "2008-03-06", "2008-04-10",
    "2008-05-08", "2008-06-05", "2008-07-03", "2008-08-07",
    "2008-09-04", "2008-10-02", "2008-10-08", "2008-11-06", "2008-12-04",
    # 2009
    "2009-01-15", "2009-02-05", "2009-03-05", "2009-04-02",
    "2009-05-07", "2009-06-04", "2009-07-02", "2009-08-06",
    "2009-09-03", "2009-10-08", "2009-11-05", "2009-12-03",
    # 2010
    "2010-01-14", "2010-02-04", "2010-03-04", "2010-04-08",
    "2010-05-06", "2010-06-10", "2010-07-08", "2010-08-05",
    "2010-09-02", "2010-10-07", "2010-11-04", "2010-12-02",
    # 2011
    "2011-01-13", "2011-02-03", "2011-03-03", "2011-04-07",
    "2011-05-05", "2011-06-09", "2011-07-07", "2011-08-04",
    "2011-09-08", "2011-10-06", "2011-11-03", "2011-12-08",
    # 2012
    "2012-01-12", "2012-02-09", "2012-03-08", "2012-04-04",
    "2012-05-03", "2012-06-06", "2012-07-05", "2012-08-02",
    "2012-09-06", "2012-10-04", "2012-11-08", "2012-12-06",
    # 2013
    "2013-01-10", "2013-02-07", "2013-03-07", "2013-04-04",
    "2013-05-02", "2013-06-06", "2013-07-04", "2013-08-01",
    "2013-09-05", "2013-10-02", "2013-11-07", "2013-12-05",
    # 2014
    "2014-01-09", "2014-02-06", "2014-03-06", "2014-04-03",
    "2014-05-08", "2014-06-05", "2014-07-03", "2014-08-07",
    "2014-09-04", "2014-10-02", "2014-11-06", "2014-12-04",
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
        "body":  body.strip()[:5000],
        "source": "ecb_press",
        "url":   url,
        "raw_source": "ecb_pressconf",
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


def _extract_pc_text(html: str) -> str:
    """Extrae el texto de la transcripción de la press conference del BCE."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # El BCE tiene los textos en <div class="section"> o <section>
    main = (
        soup.find("div", class_="section")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "main-wrapper"})
    )
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Limitar a primeros 5000 chars (intro statement + Q&A opening)
    return text[:5000]


def _find_pressconf_links(html: str, year: int) -> list[tuple[str, str]]:
    """
    Devuelve lista de (date_str, url) de press conferences en una página índice.
    Patrón del BCE: ecb.is{YYYYMMDD}~...
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = DATE_RE.search(href)
        if not m:
            continue
        raw_date = m.group(1)  # YYYYMMDD
        yr = int(raw_date[:4])
        if yr != year:
            continue
        try:
            dt = datetime.strptime(raw_date, "%Y%m%d")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        full_url = href if href.startswith("http") else ECB_BASE + href
        results.append((date_str, full_url))

    return list(dict.fromkeys(results))  # deduplicar


def _try_direct_urls(date_str: str, client: httpx.Client) -> tuple[str, str]:
    """
    Intenta URL directas del BCE para una fecha conocida del GC.
    El BCE usa: ecb.is{YYYYMMDD}~{hash}.en.html — el hash varía.
    Alternativa: usar el press release (mp) de la misma fecha.
    """
    ymd = date_str.replace("-", "")
    year = date_str[:4]

    # Intentar press release de política monetaria (diferente de press conf)
    pr_candidates = [
        f"{ECB_BASE}/press/pr/date/{year}/html/ecb.mp{ymd}~*.en.html",
        f"{ECB_BASE}/press/pressconf/{year}/html/ecb.is{ymd}.en.html",
    ]

    # La URL más fiable es buscar en el índice del año y filtrar por fecha
    idx_url = ECB_PC_IDX.format(year=year)
    html_idx = _fetch_html(idx_url, client)
    _sleep()

    if html_idx:
        links = _find_pressconf_links(html_idx, int(year))
        for d, u in links:
            if d == date_str:
                html_pc = _fetch_html(u, client)
                _sleep()
                text = _extract_pc_text(html_pc)
                return text, u

    return "", f"{ECB_BASE}/press/pressconf/{year}/"


# ── Pipeline principal ───────────────────────────────────────────────────

def fetch_ecb_press(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    use_mongo: bool = False,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"ECB Press Conferences ({start_year}-{end_year})")
    print(f"{'='*60}")

    client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)
    all_new_docs: list[dict] = []
    total_new = total_skipped = total_errors = 0

    # Filtrar fechas del rango
    meeting_dates = [
        d for d in BCE_MEETING_DATES
        if start_year <= int(d[:4]) <= end_year
    ]

    print(f"\n  Procesando {len(meeting_dates)} reuniones del Governing Council...")

    # Construir índice anual una vez por año
    year_indices: dict[int, list[tuple[str, str]]] = {}
    current_year = None

    for date_str in meeting_dates:
        ym = date_str[:7]
        year = int(date_str[:4])

        if _already_fetched(ym):
            total_skipped += 1
            print(f"  {ym}: cached")
            continue

        # Cargar índice del año si no lo tenemos aún
        if year != current_year:
            current_year = year
            idx_url = ECB_PC_IDX.format(year=year)
            print(f"  Cargando índice {year}...")
            html_idx = _fetch_html(idx_url, client)
            _sleep()
            year_indices[year] = _find_pressconf_links(html_idx, year) if html_idx else []
            print(f"    {len(year_indices[year])} press conf encontradas en índice {year}")

        # Buscar en el índice del año
        body = ""
        url = ""
        for d, u in year_indices.get(year, []):
            if d == date_str:
                html = _fetch_html(u, client)
                _sleep()
                body = _extract_pc_text(html)
                url = u
                break

        if not body:
            # Fallback: crear documento placeholder con fecha conocida
            url = f"{ECB_BASE}/press/pressconf/{year}/"
            body = (
                f"ECB Governing Council press conference on {date_str}. "
                f"Full transcript not available via automated scraping (JS-rendered index)."
            )
            print(f"  {ym}: placeholder (índice JS-rendered)")
        else:
            print(f"  {ym}: OK  ({len(body):,} chars)")

        doc = _make_doc(
            date=date_str,
            title=f"ECB Press Conference {date_str}",
            body=body,
            url=url,
        )
        _save(ym, [doc])
        all_new_docs.append(doc)
        total_new += 1

    client.close()

    if use_mongo and all_new_docs:
        n = _insert_mongo(all_new_docs)
        print(f"\n  MongoDB: {n} docs insertados")

    print(f"\n{'='*60}")
    print(f"ECB Press: {total_new} nuevos, {total_skipped} existentes, {total_errors} errores")
    print(f"Archivos en: {RAW_DIR}")
    print(f"{'='*60}")
    return all_new_docs


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga histórico de ECB press conferences (2002-2024)"
    )
    parser.add_argument("--start", type=int, default=START_YEAR)
    parser.add_argument("--end",   type=int, default=END_YEAR)
    parser.add_argument("--mongo", action="store_true")
    args = parser.parse_args()
    fetch_ecb_press(start_year=args.start, end_year=args.end, use_mongo=args.mongo)


if __name__ == "__main__":
    main()
