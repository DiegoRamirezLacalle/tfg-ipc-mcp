"""
fetch_rss_historical.py
-----------------------
Descarga historico de comunicados oficiales (2015-2024) para tres fuentes:

  BCE  — Decisiones de politica monetaria del BCE desde press releases.
  INE  — Notas de prensa mensuales del IPC via URL predecible.
  BdE  — Notas de prensa del Banco de Espana via RSS.

Cada comunicado se guarda como JSON normalizado en:
  data/raw/rss_raw/{source}/YYYY-MM.json

Es resumible: si YYYY-MM.json ya existe, se salta.
Rate limiting: 1 segundo entre peticiones HTTP.

Uso:
  python fetch_rss_historical.py                     # todas las fuentes
  python fetch_rss_historical.py --source bce        # solo BCE
  python fetch_rss_historical.py --source ine        # solo INE
  python fetch_rss_historical.py --source bde        # solo BdE
  python fetch_rss_historical.py --mongo             # ademas inserta en MongoDB

Requiere:
  pip install httpx beautifulsoup4 feedparser pymongo
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx
from bs4 import BeautifulSoup
from pymongo import MongoClient

# ── Configuracion ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BASE = PROJECT_ROOT / "data" / "raw" / "rss_raw"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "tfg_ipc_mcp"
MONGO_COLLECTION = "news_raw"

START_YEAR = 2015
END_YEAR = 2024

RATE_LIMIT = 1.0  # segundos entre peticiones

# Headers para evitar bloqueos
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TFG-IPC-MCP-Research/1.0; "
        "+academic-research)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


# ── Utilidades ────────────────────────────────────────────────
def _rate_limit():
    """Espera entre peticiones para no saturar servidores."""
    time.sleep(RATE_LIMIT)


def _save_json(source: str, year_month: str, docs: list[dict]):
    """Guarda lista de documentos como JSON en data/raw/rss_raw/{source}/YYYY-MM.json."""
    out_dir = RAW_BASE / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year_month}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    return path


def _already_fetched(source: str, year_month: str) -> bool:
    """Comprueba si ya existe el JSON para un mes dado."""
    path = RAW_BASE / source / f"{year_month}.json"
    return path.exists()


def _make_doc(
    date: str,
    title: str,
    body: str,
    source: str,
    url: str,
) -> dict:
    """Crea documento normalizado."""
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
    """Inserta documentos en MongoDB, evitando duplicados por URL."""
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
            pass  # duplicado
    return inserted


# ═══════════════════════════════════════════════════════════════
# FUENTE 1: BCE — Decisiones de politica monetaria
# ═══════════════════════════════════════════════════════════════

# Fechas de reuniones de politica monetaria del Governing Council (2015-2024).
# Fuente: calendarios oficiales del BCE publicados anualmente.
# Solo se incluyen reuniones con decision de tipos (no reuniones "non-monetary").
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

# URL base del BCE para press releases
BCE_BASE_URL = "https://www.ecb.europa.eu"
BCE_PR_INDEX = BCE_BASE_URL + "/press/pr/date/{year}/html/index.en.html"
BCE_RSS_URL = "https://www.ecb.europa.eu/rss/press.html"


def _fetch_bce_from_rss() -> list[dict]:
    """Obtiene comunicados recientes del BCE via RSS."""
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

        # Filtrar solo monetary policy
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
    """Extrae el texto completo de una pagina de press release del BCE."""
    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")

        # El contenido principal suele estar en <div class="section">
        # o en el <main> o en <article>
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
    """
    Intenta descubrir links de decisiones de politica monetaria del BCE
    para un ano dado, buscando en el indice de press releases.
    """
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
            # Buscar links de monetary policy decisions (mp en el filename)
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
    """
    Descarga historico de decisiones de politica monetaria del BCE.

    Estrategia:
    1. Intenta descubrir links desde el indice anual de press releases.
    2. Si no encuentra nada (pagina JS-rendered), usa las fechas conocidas
       de reuniones del Governing Council + RSS feed como complemento.
    """
    print(f"\n{'='*60}")
    print(f"BCE — Decisiones de politica monetaria ({start_year}-{end_year})")
    print(f"{'='*60}")

    all_docs: list[dict] = []
    total_new = 0
    client = httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True)

    # Estrategia 1: Intentar descubrir links por ano
    print("\n[1/2] Buscando links en indice anual de press releases...")
    discovered_links = []
    for year in range(start_year, end_year + 1):
        links = _discover_bce_mopo_links(year, client)
        if links:
            print(f"  {year}: {len(links)} links de monetary policy")
            discovered_links.extend([(year, l) for l in links])
        else:
            print(f"  {year}: indice no disponible (JS-rendered)")
        _rate_limit()

    # Estrategia 2: Usar fechas conocidas del Governing Council
    print("\n[2/2] Procesando fechas conocidas del Governing Council...")
    filtered_dates = [
        d for d in BCE_MEETING_DATES
        if start_year <= int(d[:4]) <= end_year
    ]
    for date_str in filtered_dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        ym = dt.strftime("%Y-%m")

        if _already_fetched("bce", ym):
            continue

        # Buscar entre los links descubiertos para este mes
        month_docs = []
        for year, link_info in discovered_links:
            if str(year) == str(dt.year):
                # Intentar descargar el press release
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

        # Si no encontramos nada via scraping, crear entrada con info conocida
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
        print(f"  {ym}: {len(month_docs)} docs ({'scraped' if month_docs[0]['body'] and 'not available' not in month_docs[0]['body'] else 'placeholder'})")

    # Complementar con RSS para comunicados recientes
    print("\n  Complementando con RSS reciente...")
    rss_docs = _fetch_bce_from_rss()
    for doc in rss_docs:
        if doc["date"]:
            ym = doc["date"][:7]
            if not _already_fetched("bce", ym):
                _save_json("bce", ym, [doc])
                all_docs.append(doc)
                total_new += 1
                print(f"  {ym}: RSS - {doc['title'][:50]}...")

    client.close()

    if use_mongo:
        n = _insert_mongo(all_docs)
        print(f"\n  MongoDB: {n} docs insertados")

    print(f"\n  BCE total: {total_new} meses nuevos")
    return all_docs


# ═══════════════════════════════════════════════════════════════
# FUENTE 2: INE — Notas de prensa del IPC
# ═══════════════════════════════════════════════════════════════

# URL predecible: https://www.ine.es/dyngs/Prensa/IPCMMYY.htm
INE_BASE_URL = "https://www.ine.es/dyngs/Prensa/IPC{mm}{yy}.htm"


def _fetch_ine_month(year: int, month: int, client: httpx.Client) -> dict | None:
    """Descarga y parsea una nota de prensa del IPC del INE."""
    yy = f"{year % 100:02d}"
    mm = f"{month:02d}"
    url = INE_BASE_URL.format(mm=mm, yy=yy)

    try:
        resp = client.get(url)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extraer titulo
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else f"IPC {month:02d}/{year}"

        # Extraer cuerpo principal
        # El INE usa varias clases: contenido, cuerpo, main-content
        content = (
            soup.find("div", class_="contenido")
            or soup.find("div", {"id": "contenido"})
            or soup.find("main")
            or soup.find("article")
        )
        if content:
            body = content.get_text(separator="\n", strip=True)
        else:
            # Fallback: todo el texto eliminando navegacion
            for nav in soup.find_all(["nav", "header", "footer", "script", "style"]):
                nav.decompose()
            body = soup.get_text(separator="\n", strip=True)

        # Limitar tamano
        body = body[:5000]

        date_str = f"{year}-{month:02d}-01"  # Primer dia del mes como referencia
        return _make_doc(
            date=date_str,
            title=title,
            body=body,
            source="ine",
            url=url,
        )
    except Exception as e:
        print(f"    [ERROR] {url}: {e}")
        return None


def fetch_ine(
    use_mongo: bool = False,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
):
    """
    Descarga historico de notas de prensa del IPC del INE.
    URL predecible: https://www.ine.es/dyngs/Prensa/IPCMMYY.htm
    """
    print(f"\n{'='*60}")
    print(f"INE — Notas de prensa del IPC ({start_year}-{end_year})")
    print(f"{'='*60}")

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
                print(f"  {ym}: OK - {body_preview}...")
            else:
                total_errors += 1
                print(f"  {ym}: no disponible")

    client.close()

    if use_mongo:
        n = _insert_mongo(all_docs)
        print(f"\n  MongoDB: {n} docs insertados")

    print(f"\n  INE total: {total_new} nuevos, {total_skipped} existentes, {total_errors} errores")
    return all_docs


# ═══════════════════════════════════════════════════════════════
# FUENTE 2b: INE PDFs — Notas de prensa IPC desde PDFs (2015-2024)
# ═══════════════════════════════════════════════════════════════

# URL confirmada: https://www.ine.es/daco/daco42/daco421/ipc{MM}{YY}.pdf
# Ejemplos: ipc0115.pdf (ene-2015), ipc1224.pdf (dic-2024)
INE_PDF_BASE = "https://www.ine.es/daco/daco42/daco421/ipc{mm}{yy}.pdf"


def _extract_ine_pdf_text(pdf_bytes: bytes) -> str:
    """Extrae texto de los PDFs de notas de prensa del IPC del INE."""
    import pdfplumber
    import io

    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[:3]:  # Primeras 3 paginas suficientes
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def _parse_ine_pdf_signals(text: str) -> dict:
    """
    Extrae senales IPC del texto del PDF.
    Los PDFs del INE contienen tasas de variacion anual y mensual
    como datos numericos directos, no necesitan LLM.
    """
    # Normalizar: comas como separador decimal en espanol
    text_norm = text.replace(",", ".")

    # Tasa de variacion anual (IPC general)
    annual_rate = None
    for pattern in [
        r"variaci[oó]n\s+anual[^\d-]*(-?\d+\.?\d*)",
        r"tasa\s+anual[^\d-]*(-?\d+\.?\d*)",
        r"IPC[^\d-]*(-?\d+\.?\d*)\s*(?:por\s*ciento|%)",
        r"(-?\d+\.?\d*)\s*(?:por\s*ciento|%)[^\n]*anual",
    ]:
        m = re.search(pattern, text_norm, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if -5.0 <= val <= 15.0:  # rango razonable para IPC espanol
                    annual_rate = val
                    break
            except ValueError:
                continue

    # Tasa de variacion mensual
    monthly_rate = None
    for pattern in [
        r"variaci[oó]n\s+mensual[^\d-]*(-?\d+\.?\d*)",
        r"tasa\s+mensual[^\d-]*(-?\d+\.?\d*)",
        r"(-?\d+\.?\d*)\s*(?:por\s*ciento|%)[^\n]*mensual",
    ]:
        m = re.search(pattern, text_norm, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if -3.0 <= val <= 3.0:  # rango razonable mensual
                    monthly_rate = val
                    break
            except ValueError:
                continue

    # Construir senales
    if annual_rate is None and monthly_rate is None:
        # No se pudieron extraer tasas — devolver defaults neutros
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

    # Decision basada en cambio mensual
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

    # Tono basado en nivel de inflacion anual
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
        "uncertainty_index": 0.3,  # datos IPC son objetivos, baja incertidumbre
        "topic": "inflacion",
        "ipc_general": annual_rate,
        "ipc_monthly": monthly_rate,
    }


def fetch_ine_pdfs(
    use_mongo: bool = True,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
):
    """
    Descarga 120 PDFs de notas de prensa del IPC del INE (2015-2024).

    URL: https://www.ine.es/daco/daco42/daco421/ipc{MM}{YY}.pdf
    Extrae tasas de variacion con regex (sin LLM).
    Guarda en MongoDB con source=ine, raw_source=pdf_historical, processed=True.
    Resumible: salta meses ya existentes en MongoDB.
    """
    import pdfplumber as _pdf_check  # verify availability at startup
    del _pdf_check

    print(f"\n{'='*60}")
    print(f"INE PDFs — Notas de prensa IPC ({start_year}-{end_year})")
    print(f"{'='*60}")

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

            # Resumible: saltar si ya existe en MongoDB
            if col.count_documents({"url": url}) > 0:
                total_skipped += 1
                continue

            try:
                resp = client_http.get(url)
                _rate_limit()

                if resp.status_code != 200:
                    print(f"  {ym}: HTTP {resp.status_code} — no disponible")
                    total_errors += 1
                    continue

                if len(resp.content) < 1000:
                    print(f"  {ym}: respuesta demasiado pequena ({len(resp.content)}b)")
                    total_errors += 1
                    continue

                # Extraer texto del PDF
                text = _extract_ine_pdf_text(resp.content)
                if not text or len(text) < 50:
                    print(f"  {ym}: PDF sin texto extraible")
                    total_errors += 1
                    continue

                # Parsear senales directamente (datos numericos, sin LLM)
                signals = _parse_ine_pdf_signals(text)

                ipc_str = (f"IPC anual={signals['ipc_general']}%"
                           if signals['ipc_general'] is not None
                           else "tasa no parseada")

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
                        pass  # duplicado

                # Guardar en JSON local (excluir _id que MongoDB añade al dict)
                doc_json = {k: v for k, v in doc.items() if k != "_id"}
                _save_json("ine_pdf", ym, [doc_json])

                total_new += 1
                print(f"  {ym}: OK — {ipc_str} | {signals['tone']} | shock={signals['shock_score']}")

            except Exception as e:
                print(f"  {ym}: ERROR — {e}")
                total_errors += 1
                _rate_limit()  # esperar aunque haya error

    client_http.close()
    mongo_client.close()

    print(f"\n  INE PDFs: {total_new} nuevos, {total_skipped} existentes, {total_errors} errores")
    return total_new


# ═══════════════════════════════════════════════════════════════
# FUENTE 3: BdE — Banco de Espana (notas de prensa)
# ═══════════════════════════════════════════════════════════════

BDE_RSS_URL = "https://www.bde.es/rss/es/"

# Palabras clave para filtrar notas relevantes sobre tipos e inflacion
BDE_KEYWORDS = [
    "tipo", "interes", "inflacion", "ipc", "precio",
    "politica monetaria", "euribor", "credito", "hipoteca",
    "estabilidad", "financiera", "supervision", "bancaria",
]


def fetch_bde(use_mongo: bool = False):
    """
    Descarga notas de prensa del Banco de Espana via RSS.

    Limitacion conocida: el RSS del BdE solo retiene los ultimos ~20-50
    articulos. Para historico completo seria necesario scraping con
    Selenium/Playwright (la web del BdE usa JS rendering pesado).
    Esto se documenta como limitacion del TFG.
    """
    print(f"\n{'='*60}")
    print("BdE — Notas de prensa del Banco de Espana (RSS)")
    print(f"{'='*60}")

    all_docs: list[dict] = []

    # Intentar multiples URLs del RSS del BdE
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
                print(f"  RSS {rss_url}: {len(feed.entries)} entries")
        except Exception as e:
            print(f"  RSS {rss_url}: error - {e}")

    if not entries:
        print("  WARN: No se obtuvieron entradas del BdE RSS.")
        print("  Limitacion conocida: el sitio del BdE usa JS rendering.")
        print("  Para historico completo se requiere Selenium/Playwright.")
        return all_docs

    # Procesar entradas
    seen_months: set[str] = set()
    for entry in entries:
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "")

        # Filtrar por relevancia
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

        # Agrupar por mes
        if ym not in seen_months:
            seen_months.add(ym)
            if not _already_fetched("bde", ym):
                _save_json("bde", ym, [doc])
                all_docs.append(doc)
                print(f"  {ym}: {title[:60]}...")

    if use_mongo:
        n = _insert_mongo(all_docs)
        print(f"\n  MongoDB: {n} docs insertados")

    print(f"\n  BdE total: {len(all_docs)} meses nuevos (limitado por RSS)")
    return all_docs


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Descarga historico de comunicados oficiales (2015-2024)"
    )
    parser.add_argument(
        "--source",
        choices=["bce", "ine", "ine-pdf", "bde"],
        default=None,
        help="Solo descargar una fuente (default: todas). 'ine-pdf' descarga PDFs historicos 2015-2024.",
    )
    parser.add_argument(
        "--mongo",
        action="store_true",
        help="Insertar documentos en MongoDB ademas de guardar JSON",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"Ano inicio (default: {START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"Ano fin (default: {END_YEAR})",
    )

    args = parser.parse_args()

    start_y = args.start_year
    end_y = args.end_year
    sources = [args.source] if args.source else ["bce", "ine-pdf", "bde"]

    print(f"Descarga historica de comunicados oficiales")
    print(f"Rango: {start_y}-{end_y}")
    print(f"Fuentes: {', '.join(sources)}")
    print(f"Rate limit: {RATE_LIMIT}s entre peticiones")
    print(f"Output: {RAW_BASE}/")

    if "bce" in sources:
        fetch_bce(use_mongo=args.mongo, start_year=start_y, end_year=end_y)

    if "ine" in sources:
        fetch_ine(use_mongo=args.mongo, start_year=start_y, end_year=end_y)

    if "ine-pdf" in sources:
        fetch_ine_pdfs(use_mongo=True, start_year=start_y, end_year=end_y)

    if "bde" in sources:
        fetch_bde(use_mongo=args.mongo)

    print(f"\n{'='*60}")
    print("Descarga completada.")
    print(f"Archivos en: {RAW_BASE}")
    if args.mongo:
        print(f"Documentos insertados en MongoDB: {MONGO_DB}.{MONGO_COLLECTION}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
