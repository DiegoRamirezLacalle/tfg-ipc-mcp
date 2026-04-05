"""
mcp_server.py
-------------
Servidor MCP (FastMCP, transporte stdio) con 5 herramientas:

  1. fetch_gdelt_spain   - GDELT v2 cuantitativo (sin LLM)
  2. fetch_rss_official   - RSS de BCE / INE / BdE
  3. search_news          - Busqueda unificada en MongoDB
  4. get_macro_news       - Noticias macro (GDELT + RSS)
  5. get_entity_news      - Noticias por entidad (BCE, INE, BdE)

Almacenamiento: MongoDB (coleccion news_raw) + parquet versionado.

Requiere:
    pip install "mcp[cli]" httpx pandas pymongo feedparser
    mongod corriendo en localhost:27017
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient

# ── Configuracion ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gdelt_spain_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "tfg_ipc_mcp"
MONGO_COLLECTION = "news_raw"

# GDELT v2
GDELT_EVENTS_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.export.CSV.zip"
GDELT_COUNTRY_CODE = "SP"
EVENT_CODE_MIN = 100
EVENT_CODE_MAX = 199
TOP_N_PER_DAY = 100

# RSS feeds oficiales
RSS_FEEDS = {
    "bce": "https://www.ecb.europa.eu/rss/press.html",
    "ine": "https://www.ine.es/rss/rss_notas_prensa.xml",
    "bde": "https://www.bde.es/rss/es/",
}

# Columnas GDELT v2 export (61 columnas)
GDELT_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code",
    "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code",
    "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]


# ── MongoDB helper ────────────────────────────────────────────
def _get_collection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_COLLECTION]


# ── GDELT helpers ─────────────────────────────────────────────
def _gdelt_timestamps_for_month(year: int, month: int) -> list[str]:
    """4 snapshots/dia (00, 06, 12, 18h) para todo el mes."""
    from calendar import monthrange
    _, last_day = monthrange(year, month)
    timestamps = []
    for day in range(1, last_day + 1):
        for hour in (0, 6, 12, 18):
            timestamps.append(f"{year:04d}{month:02d}{day:02d}{hour:02d}0000")
    return timestamps


def _download_gdelt_month(year: int, month: int) -> pd.DataFrame:
    """Descarga GDELT v2 de un mes, filtra Espana + EventCode 100-199."""
    timestamps = _gdelt_timestamps_for_month(year, month)
    frames: list[pd.DataFrame] = []

    for ts in timestamps:
        url = GDELT_EVENTS_URL.format(ts=ts)
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f, sep="\t", header=None, names=GDELT_COLS,
                        dtype=str, on_bad_lines="skip",
                    )
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Filtro Espana
    spain_mask = (
        (df["Actor1CountryCode"] == GDELT_COUNTRY_CODE)
        | (df["Actor2CountryCode"] == GDELT_COUNTRY_CODE)
        | (df["ActionGeo_CountryCode"] == GDELT_COUNTRY_CODE)
    )
    df = df[spain_mask].copy()

    # Filtro EventCode 100-199
    df["EventCode_int"] = pd.to_numeric(df["EventCode"], errors="coerce")
    df = df[
        (df["EventCode_int"] >= EVENT_CODE_MIN)
        & (df["EventCode_int"] <= EVENT_CODE_MAX)
    ]

    # Deduplicar + top N por dia
    df = df.drop_duplicates(subset=["GlobalEventID"])
    df["NumArticles_f"] = pd.to_numeric(df["NumArticles"], errors="coerce").fillna(0)
    df["Day_str"] = df["Day"].astype(str)
    df = (
        df.sort_values("NumArticles_f", ascending=False)
        .groupby("Day_str")
        .head(TOP_N_PER_DAY)
        .reset_index(drop=True)
    )
    return df


def _aggregate_gdelt_monthly(df: pd.DataFrame) -> dict:
    """Agrega senales cuantitativas GDELT a nivel mensual."""
    df["AvgTone_f"] = pd.to_numeric(df["AvgTone"], errors="coerce")
    df["GoldsteinScale_f"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["NumArticles_f"] = pd.to_numeric(df["NumArticles"], errors="coerce")

    return {
        "gdelt_avg_tone": round(float(df["AvgTone_f"].mean()), 4),
        "gdelt_goldstein": round(float(df["GoldsteinScale_f"].mean()), 4),
        "gdelt_n_articles": int(df["NumArticles_f"].sum()),
        "n_events": len(df),
    }


# ── RSS helpers ───────────────────────────────────────────────
def _parse_rss_feed(source: str, start_date: str, end_date: str) -> list[dict]:
    """
    Descarga y parsea RSS. Guarda todo lo disponible en el feed (los feeds
    RSS solo retienen los ultimos 20-100 articulos, no se puede filtrar
    historico). El filtro temporal se aplica en news_to_features.py
    usando ingestion_timestamp.
    """
    feed_url = RSS_FEEDS.get(source)
    if not feed_url:
        return []

    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        return []

    now = datetime.now(timezone.utc)
    results = []

    for entry in feed.entries:
        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except Exception:
                pass
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                pub_date = datetime(*entry.updated_parsed[:6])
            except Exception:
                pass

        title = getattr(entry, "title", "")
        body = getattr(entry, "summary", "") or getattr(entry, "description", "")
        link = getattr(entry, "link", "")

        doc = {
            "date": pub_date.strftime("%Y-%m-%d") if pub_date else None,
            "title": title,
            "body": body,
            "source": source,
            "url": link,
            "raw_source": "rss",
            "ingestion_timestamp": now.isoformat(),
            "processed": False,
        }
        results.append(doc)

    return results


def _store_in_mongo(docs: list[dict]) -> int:
    """Guarda documentos en MongoDB, evitando duplicados por URL."""
    if not docs:
        return 0

    col = _get_collection()
    col.create_index("url", unique=True, sparse=True)

    inserted = 0
    for doc in docs:
        try:
            col.insert_one(doc)
            inserted += 1
        except Exception:
            # Duplicado por URL — skip
            continue
    return inserted


# ── Servidor MCP ──────────────────────────────────────────────
mcp = FastMCP(name="tfg-ipc-mcp-news")


@mcp.tool()
def fetch_gdelt_spain(start_date: str, end_date: str) -> str:
    """
    Descarga GDELT v2 filtrado por Espana (EventCode 100-199).
    Devuelve senales cuantitativas agregadas por mes.
    Guarda raw en data/raw/gdelt_spain_raw/ y en MongoDB.
    No pasa por LLM.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    results = []
    current = start.replace(day=1)
    while current <= end:
        ym = f"{current.year:04d}-{current.month:02d}"
        raw_path = RAW_DIR / f"{ym}.parquet"

        # Cache check
        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            agg = _aggregate_gdelt_monthly(df) if not df.empty else {
                "gdelt_avg_tone": 0.0, "gdelt_goldstein": 0.0,
                "gdelt_n_articles": 0, "n_events": 0,
            }
            agg["year_month"] = ym
            agg["status"] = "cached"
            results.append(agg)
        else:
            df = _download_gdelt_month(current.year, current.month)
            if not df.empty:
                df.to_parquet(raw_path, index=False)
                agg = _aggregate_gdelt_monthly(df)
                # Guardar en MongoDB como fuente gdelt
                now = datetime.now(timezone.utc)
                mongo_doc = {
                    "date": ym,
                    "title": f"GDELT Spain aggregate {ym}",
                    "body": json.dumps(agg),
                    "source": "gdelt",
                    "url": f"gdelt://{ym}",
                    "raw_source": "gdelt_v2",
                    "ingestion_timestamp": now.isoformat(),
                    "processed": True,  # Ya es cuantitativo, no necesita LLM
                    "signals": agg,
                }
                _store_in_mongo([mongo_doc])
            else:
                agg = {
                    "gdelt_avg_tone": 0.0, "gdelt_goldstein": 0.0,
                    "gdelt_n_articles": 0, "n_events": 0,
                }
            agg["year_month"] = ym
            agg["status"] = "downloaded"
            results.append(agg)

        # Siguiente mes
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return json.dumps(results)


@mcp.tool()
def fetch_rss_official(source: str, start_date: str, end_date: str) -> str:
    """
    Descarga comunicados RSS de una fuente oficial (bce|ine|bde).
    Guarda textos normalizados en MongoDB con processed=false.
    Devuelve resumen de lo descargado.
    """
    if source not in RSS_FEEDS:
        return json.dumps({
            "error": f"Fuente no valida: {source}. Usar: {list(RSS_FEEDS.keys())}"
        })

    docs = _parse_rss_feed(source, start_date, end_date)
    inserted = _store_in_mongo(docs)

    return json.dumps({
        "source": source,
        "feed_url": RSS_FEEDS[source],
        "articles_found": len(docs),
        "articles_inserted": inserted,
        "date_range": {"start": start_date, "end": end_date},
    })


@mcp.tool()
def search_news(query: str, start_date: str, end_date: str) -> str:
    """
    Busca en todas las fuentes almacenadas en MongoDB.
    Filtra por texto (titulo/body) y rango de fechas.
    """
    col = _get_collection()

    mongo_filter = {
        "date": {"$gte": start_date, "$lte": end_date},
        "$or": [
            {"title": {"$regex": query, "$options": "i"}},
            {"body": {"$regex": query, "$options": "i"}},
        ],
    }

    docs = list(col.find(mongo_filter, {"_id": 0}).sort("date", -1).limit(100))
    return json.dumps(docs, default=str)


@mcp.tool()
def get_macro_news(topic: str, country: str, start_date: str, end_date: str) -> str:
    """
    Noticias macroeconomicas: busca en GDELT + RSS bancos centrales.
    Topics: tipos_interes, inflacion, empleo, pib.
    """
    col = _get_collection()

    # Buscar en MongoDB por topic en signals o en texto
    topic_keywords = {
        "tipos_interes": "tipos|interes|rates|rate|monetary",
        "inflacion": "inflacion|inflation|IPC|CPI|precios|prices",
        "empleo": "empleo|employment|unemployment|paro|desempleo",
        "pib": "PIB|GDP|crecimiento|growth|recesion|recession",
    }
    keyword_pattern = topic_keywords.get(topic, topic)

    mongo_filter = {
        "date": {"$gte": start_date, "$lte": end_date},
        "$or": [
            {"title": {"$regex": keyword_pattern, "$options": "i"}},
            {"body": {"$regex": keyword_pattern, "$options": "i"}},
            {"signals.topic": topic},
        ],
    }

    docs = list(col.find(mongo_filter, {"_id": 0}).sort("date", -1).limit(100))
    return json.dumps(docs, default=str)


@mcp.tool()
def get_entity_news(entity: str, start_date: str, end_date: str) -> str:
    """
    Noticias sobre una entidad concreta: bce, ine, bde.
    """
    entity_lower = entity.lower()
    col = _get_collection()

    mongo_filter = {
        "date": {"$gte": start_date, "$lte": end_date},
        "source": entity_lower,
    }

    docs = list(col.find(mongo_filter, {"_id": 0}).sort("date", -1).limit(100))
    return json.dumps(docs, default=str)


# ── Punto de entrada ──────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
