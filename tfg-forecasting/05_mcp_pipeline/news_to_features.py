"""
news_to_features.py
-------------------
Orquestador del pipeline MCP completo:

  --acquire     Descarga GDELT + RSS y almacena en MongoDB
  --process     Extrae senales LLM de comunicados RSS no procesados
  --build-c1    Agrega a frecuencia mensual y exporta news_signals.parquet

Principio clave: SEPARACION TOTAL entre adquisicion (Internet) y ejecucion.
Control de leakage: ingestion_timestamp < t para cada origen de backtesting.

Esquema final news_signals.parquet:
  date | gdelt_avg_tone | gdelt_goldstein | gdelt_n_articles |
       | bce_shock_score | bce_uncertainty | bce_tone |
       | ine_surprise_score | ine_topic | dominant_topic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

# ── Rutas ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
SIGNALS_PATH = DATA_PROC / "news_signals.parquet"
FEATURES_EXOG_PATH = DATA_PROC / "features_exog.parquet"
FEATURES_C1_PATH = DATA_PROC / "features_c1.parquet"

sys.path.insert(0, str(PROJECT_ROOT.parent / "shared"))
from constants import FREQ

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "tfg_ipc_mcp"
MONGO_COLLECTION = "news_raw"

# Rango historico para GDELT (spec: desde 2015)
GDELT_START = "2015-01-01"
GDELT_END = "2024-12-31"

# Fuentes RSS
RSS_SOURCES = ["bce", "ine", "bde"]


def _get_collection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_COLLECTION]


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Genera lista de (year, month) entre dos fechas mensuales."""
    periods = pd.date_range(start=start, end=end, freq=FREQ)
    return [(d.year, d.month) for d in periods]


# ── Paso 1: Adquisicion ──────────────────────────────────────
def acquire(start_date: str = GDELT_START, end_date: str = GDELT_END):
    """
    Descarga GDELT cuantitativo + RSS oficiales via MCP client.
    Almacena todo en MongoDB.
    """
    from mcp_client import MCPPipeline

    months = _month_range(start_date, end_date)

    with MCPPipeline(timeout=300) as pipeline:
        # 1a. GDELT mes a mes (cada llamada ~2 min, dentro del timeout de 300s)
        print(f"\n[acquire] GDELT v2: {start_date} - {end_date} ({len(months)} meses)")
        cached, downloaded = 0, 0
        for i, (year, month) in enumerate(months):
            ym_start = f"{year:04d}-{month:02d}-01"
            # Ultimo dia del mes
            from calendar import monthrange
            last_day = monthrange(year, month)[1]
            ym_end = f"{year:04d}-{month:02d}-{last_day:02d}"

            results = pipeline.fetch_gdelt(ym_start, ym_end)
            status = results[0].get("status") if results else "error"
            if status == "cached":
                cached += 1
            else:
                downloaded += 1
            print(f"  [{i+1}/{len(months)}] {year}-{month:02d}: {status} "
                  f"({results[0].get('n_events', 0)} eventos)")

        print(f"  Total: {cached} cached, {downloaded} nuevos")

        # 1b. RSS de cada fuente
        for source in RSS_SOURCES:
            print(f"\n[acquire] RSS {source.upper()}: {start_date} - {end_date}")
            result = pipeline.fetch_rss(source, start_date, end_date)
            print(f"  Encontrados: {result.get('articles_found', 0)}")
            print(f"  Insertados:  {result.get('articles_inserted', 0)}")

    col = _get_collection()
    total = col.count_documents({})
    unprocessed = col.count_documents({"processed": False})
    print(f"\n[acquire] MongoDB total: {total} docs ({unprocessed} sin procesar)")


# ── Paso 2: Procesamiento LLM ────────────────────────────────
def process_rss():
    """
    Busca comunicados RSS sin procesar en MongoDB,
    extrae senales con Qwen3:4b, actualiza el documento.
    """
    from agent_extractor import extract_signals

    col = _get_collection()
    pending = list(col.find({
        "processed": False,
        "raw_source": {"$in": ["rss", "rss_historical"]},
    }))

    if not pending:
        print("[process] No hay comunicados RSS pendientes.")
        return

    print(f"[process] {len(pending)} comunicados por procesar con qwen3:4b")

    for i, doc in enumerate(pending):
        text = f"{doc.get('title', '')} {doc.get('body', '')}"
        source = doc.get("source", "")
        print(f"  [{i+1}/{len(pending)}] {source}: {doc.get('title', '')[:60]}...")

        signals = extract_signals(text, source=source)

        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"processed": True, "signals": signals}},
        )

    print(f"[process] Completado. {len(pending)} comunicados procesados.")


# ── Paso 3: Construccion del parquet ──────────────────────────
def build_c1():
    """
    Lee MongoDB, agrega a frecuencia mensual, exporta news_signals.parquet.
    Genera features_c1.parquet = features_exog + news_signals.
    """
    col = _get_collection()

    # ── GDELT: senales cuantitativas ──
    gdelt_docs = list(col.find({"raw_source": "gdelt_v2", "processed": True}))
    gdelt_rows = []
    for doc in gdelt_docs:
        signals = doc.get("signals", {})
        gdelt_rows.append({
            "date": doc["date"],  # "YYYY-MM"
            "gdelt_avg_tone": signals.get("gdelt_avg_tone", 0.0),
            "gdelt_goldstein": signals.get("gdelt_goldstein", 0.0),
            "gdelt_n_articles": signals.get("gdelt_n_articles", 0),
        })

    df_gdelt = pd.DataFrame(gdelt_rows) if gdelt_rows else pd.DataFrame(
        columns=["date", "gdelt_avg_tone", "gdelt_goldstein", "gdelt_n_articles"]
    )

    # ── RSS + PDF: senales por fuente y mes ──
    # Incluye: rss (tiempo real), rss_historical (scraping), pdf_historical (INE PDFs)
    rss_docs = list(col.find({
        "raw_source": {"$in": ["rss", "rss_historical", "pdf_historical"]},
        "processed": True,
    }))

    bce_rows, ine_rows = [], []
    for doc in rss_docs:
        signals = doc.get("signals", {})
        date_str = doc.get("date", "")
        if not date_str or len(date_str) < 7:
            continue
        ym = date_str[:7]  # "YYYY-MM"
        source = doc.get("source", "")

        if source == "bce":
            bce_rows.append({
                "date": ym,
                "shock_score": signals.get("shock_score", 0.0),
                "uncertainty_index": signals.get("uncertainty_index", 0.5),
                "tone": signals.get("tone", "neutral"),
            })
        elif source == "ine":
            ine_rows.append({
                "date": ym,
                "shock_score": signals.get("shock_score", 0.0),
                "topic": signals.get("topic", "otro"),
            })

    # Agregar BCE mensual (media de scores, moda de tone)
    df_bce_agg = pd.DataFrame(columns=["date", "bce_shock_score", "bce_uncertainty", "bce_tone"])
    if bce_rows:
        df_bce = pd.DataFrame(bce_rows)
        tone_map = {"hawkish": 1, "neutral": 0, "dovish": -1, "positivo": 0.5, "negativo": -0.5}

        def _agg_bce(grp):
            tones = grp["tone"].tolist()
            most_common = Counter(tones).most_common(1)[0][0]
            return pd.Series({
                "bce_shock_score": round(grp["shock_score"].mean(), 2),
                "bce_uncertainty": round(grp["uncertainty_index"].mean(), 2),
                "bce_tone": most_common,
            })

        df_bce_agg = df_bce.groupby("date").apply(_agg_bce, include_groups=False).reset_index()

    # Agregar INE mensual
    df_ine_agg = pd.DataFrame(columns=["date", "ine_surprise_score", "ine_topic"])
    if ine_rows:
        df_ine = pd.DataFrame(ine_rows)

        def _agg_ine(grp):
            topics = grp["topic"].tolist()
            most_common = Counter(topics).most_common(1)[0][0]
            return pd.Series({
                "ine_surprise_score": round(grp["shock_score"].mean(), 2),
                "ine_topic": most_common,
            })

        df_ine_agg = df_ine.groupby("date").apply(_agg_ine, include_groups=False).reset_index()

    # ── Merge todo por fecha ──
    # Generar indice de meses completo
    all_months = pd.date_range(start=GDELT_START, end=GDELT_END, freq=FREQ)
    df_base = pd.DataFrame({"date": all_months.strftime("%Y-%m")})

    df_signals = df_base.copy()
    if not df_gdelt.empty:
        df_signals = df_signals.merge(df_gdelt, on="date", how="left")
    if not df_bce_agg.empty:
        df_signals = df_signals.merge(df_bce_agg, on="date", how="left")
    if not df_ine_agg.empty:
        df_signals = df_signals.merge(df_ine_agg, on="date", how="left")

    # Rellenar NaN
    fill = {
        "gdelt_avg_tone": 0.0,
        "gdelt_goldstein": 0.0,
        "gdelt_n_articles": 0,
        "bce_shock_score": 0.0,
        "bce_uncertainty": 0.5,
        "bce_tone": "neutral",
        "ine_surprise_score": 0.0,
        "ine_topic": "otro",
    }
    for col_name, val in fill.items():
        if col_name not in df_signals.columns:
            df_signals[col_name] = val
        else:
            df_signals[col_name] = df_signals[col_name].fillna(val)

    df_signals["gdelt_n_articles"] = df_signals["gdelt_n_articles"].astype(int)

    # ── FIX dominant_topic: captura meses de politica monetaria estable ──
    # bce_shock_score > 0 indica que hubo reunion y decision relevante,
    # aunque el tono sea neutral (sin cambio de tipos pero con comunicado)
    def _dominant(row):
        if row["bce_tone"] in ("hawkish", "dovish") or row["bce_shock_score"] > 0:
            return "tipos_interes"
        if row["ine_topic"] != "otro":
            return row["ine_topic"]
        return "otro"

    df_signals["dominant_topic"] = df_signals.apply(_dominant, axis=1)

    # Convertir date a datetime para alinear con IPC
    df_signals["date"] = pd.to_datetime(df_signals["date"] + "-01")

    # Guardar news_signals.parquet (sin shift: contiene senales del mes t)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df_signals.to_parquet(SIGNALS_PATH, index=False)
    print(f"[build-c1] news_signals.parquet: {len(df_signals)} filas")
    print(f"           Columnas: {list(df_signals.columns)}")

    # ── Merge con features_exog para features_c1 ──
    if not FEATURES_EXOG_PATH.exists():
        print("[build-c1] WARN: features_exog.parquet no existe. Solo se genero news_signals.")
        return

    df_exog = pd.read_parquet(FEATURES_EXOG_PATH)
    if "date" not in df_exog.columns:
        # La fecha esta en el indice (con o sin nombre)
        df_exog = df_exog.reset_index()
        df_exog = df_exog.rename(columns={df_exog.columns[0]: "date"})
    df_exog["date"] = pd.to_datetime(df_exog["date"])

    # ── FIX LEAKAGE: shift +1 mes sobre todas las senales ──
    # Las senales del mes t (GDELT, BCE, INE) no estan disponibles
    # hasta despues de que acaba el mes t. Al predecir IPC(t) con
    # origen en t, solo podemos usar senales de t-1.
    # El shift mueve la senal del mes t a la fila del mes t+1,
    # de modo que features_c1[t] contiene senales de t-1.
    signal_cols = [c for c in df_signals.columns if c != "date"]
    df_signals_lagged = df_signals.copy()
    df_signals_lagged[signal_cols] = df_signals_lagged[signal_cols].shift(1)
    # La primera fila queda NaN tras el shift -> rellenar con defaults
    for col_name, val in fill.items():
        if col_name in df_signals_lagged.columns:
            df_signals_lagged[col_name] = df_signals_lagged[col_name].fillna(val)
    # dominant_topic tambien queda NaN en la primera fila
    df_signals_lagged["dominant_topic"] = df_signals_lagged["dominant_topic"].fillna("otro")

    df_c1 = df_exog.merge(
        df_signals_lagged[["date"] + signal_cols],
        on="date",
        how="left",
    )

    # Rellenar NaN de meses fuera del rango de senales con defaults
    for col_name, val in fill.items():
        if col_name in df_c1.columns:
            df_c1[col_name] = df_c1[col_name].fillna(val)
    if "dominant_topic" in df_c1.columns:
        df_c1["dominant_topic"] = df_c1["dominant_topic"].fillna("otro")

    df_c1.to_parquet(FEATURES_C1_PATH, index=False)
    print(f"\n[build-c1] features_c1.parquet: {len(df_c1)} filas, {len(df_c1.columns)} cols")
    print(f"           [leakage fix] senales shifteadas +1 mes (senales[t] en fila[t+1])")
    print(f"           Columnas: {list(df_c1.columns)}")


# ── CLI ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline MCP: adquisicion, procesamiento LLM y features C1"
    )
    parser.add_argument(
        "--acquire", action="store_true",
        help="Descargar GDELT + RSS y almacenar en MongoDB (requiere Internet)",
    )
    parser.add_argument(
        "--process", action="store_true",
        help="Procesar comunicados RSS pendientes con Qwen3:4b (requiere Ollama)",
    )
    parser.add_argument(
        "--build-c1", action="store_true",
        help="Agregar senales mensuales y exportar parquet",
    )
    parser.add_argument(
        "--start", type=str, default=GDELT_START,
        help=f"Fecha inicio (default: {GDELT_START})",
    )
    parser.add_argument(
        "--end", type=str, default=GDELT_END,
        help=f"Fecha fin (default: {GDELT_END})",
    )

    args = parser.parse_args()

    if not args.acquire and not args.process and not args.build_c1:
        parser.print_help()
        sys.exit(0)

    if args.acquire:
        acquire(start_date=args.start, end_date=args.end)

    if args.process:
        process_rss()

    if args.build_c1:
        build_c1()


if __name__ == "__main__":
    main()
