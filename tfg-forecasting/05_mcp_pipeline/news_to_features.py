"""MCP pipeline orchestrator: acquire data, process with LLM, build C1 features.

  --acquire     Download GDELT + RSS and store in MongoDB
  --process     Extract LLM signals from unprocessed RSS releases
  --build-c1    Aggregate to monthly frequency and export news_signals.parquet

Leakage control: all news signals are shifted +1 month before merging with exog features.

Output schema (news_signals.parquet):
  date | gdelt_avg_tone | gdelt_goldstein | gdelt_n_articles |
       | bce_shock_score | bce_uncertainty | bce_tone |
       | ine_surprise_score | ine_topic | dominant_topic
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
SIGNALS_PATH = DATA_PROC / "news_signals.parquet"
FEATURES_EXOG_PATH = DATA_PROC / "features_exog.parquet"
FEATURES_C1_PATH = DATA_PROC / "features_c1.parquet"

sys.path.insert(0, str(PROJECT_ROOT.parent))

from shared.constants import FREQ
from shared.logger import get_logger

logger = get_logger(__name__)

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "tfg_ipc_mcp"
MONGO_COLLECTION = "news_raw"

GDELT_START = "2015-01-01"
GDELT_END = "2024-12-31"

RSS_SOURCES = ["bce", "ine", "bde"]


def _get_collection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_COLLECTION]


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Return list of (year, month) between two monthly dates."""
    periods = pd.date_range(start=start, end=end, freq=FREQ)
    return [(d.year, d.month) for d in periods]


# Step 1: Acquire

def acquire(start_date: str = GDELT_START, end_date: str = GDELT_END):
    """Download GDELT (quantitative) + official RSS via MCP client and store in MongoDB."""
    from mcp_client import MCPPipeline

    months = _month_range(start_date, end_date)

    with MCPPipeline(timeout=300) as pipeline:
        logger.info(f"[acquire] GDELT v2: {start_date} - {end_date} ({len(months)} months)")
        cached, downloaded = 0, 0
        for i, (year, month) in enumerate(months):
            ym_start = f"{year:04d}-{month:02d}-01"
            from calendar import monthrange
            last_day = monthrange(year, month)[1]
            ym_end = f"{year:04d}-{month:02d}-{last_day:02d}"

            results = pipeline.fetch_gdelt(ym_start, ym_end)
            status = results[0].get("status") if results else "error"
            if status == "cached":
                cached += 1
            else:
                downloaded += 1
            logger.info(
                f"  [{i+1}/{len(months)}] {year}-{month:02d}: {status} "
                f"({results[0].get('n_events', 0)} events)"
            )

        logger.info(f"  Total: {cached} cached, {downloaded} new")

        for source in RSS_SOURCES:
            logger.info(f"[acquire] RSS {source.upper()}: {start_date} - {end_date}")
            result = pipeline.fetch_rss(source, start_date, end_date)
            logger.info(f"  Found:    {result.get('articles_found', 0)}")
            logger.info(f"  Inserted: {result.get('articles_inserted', 0)}")

    col = _get_collection()
    total = col.count_documents({})
    unprocessed = col.count_documents({"processed": False})
    logger.info(f"[acquire] MongoDB total: {total} docs ({unprocessed} unprocessed)")


# Step 2: LLM processing

def process_rss():
    """Find unprocessed RSS releases in MongoDB and extract signals with Qwen3:4b."""
    from agent_extractor import extract_signals

    col = _get_collection()
    pending = list(col.find({
        "processed": False,
        "raw_source": {"$in": ["rss", "rss_historical"]},
    }))

    if not pending:
        logger.info("[process] No pending RSS releases.")
        return

    logger.info(f"[process] {len(pending)} releases to process with qwen3:4b")

    for i, doc in enumerate(pending):
        text = f"{doc.get('title', '')} {doc.get('body', '')}"
        source = doc.get("source", "")
        logger.info(f"  [{i+1}/{len(pending)}] {source}: {doc.get('title', '')[:60]}...")

        signals = extract_signals(text, source=source)

        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"processed": True, "signals": signals}},
        )

    logger.info(f"[process] Done. {len(pending)} releases processed.")


# Step 3: Build parquet

def build_c1():
    """Read MongoDB, aggregate to monthly frequency, export news_signals.parquet and features_c1.parquet."""
    col = _get_collection()

    # GDELT: quantitative signals
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

    # RSS + PDF signals by source and month
    # Includes: rss (real-time), rss_historical (scraped), pdf_historical (INE PDFs)
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

    # Aggregate BCE monthly (mean scores, mode of tone)
    df_bce_agg = pd.DataFrame(columns=["date", "bce_shock_score", "bce_uncertainty", "bce_tone"])
    if bce_rows:
        df_bce = pd.DataFrame(bce_rows)

        def _agg_bce(grp):
            tones = grp["tone"].tolist()
            most_common = Counter(tones).most_common(1)[0][0]
            return pd.Series({
                "bce_shock_score": round(grp["shock_score"].mean(), 2),
                "bce_uncertainty": round(grp["uncertainty_index"].mean(), 2),
                "bce_tone": most_common,
            })

        df_bce_agg = df_bce.groupby("date").apply(_agg_bce, include_groups=False).reset_index()

    # Aggregate INE monthly
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

    # Merge all by date
    all_months = pd.date_range(start=GDELT_START, end=GDELT_END, freq=FREQ)
    df_base = pd.DataFrame({"date": all_months.strftime("%Y-%m")})

    df_signals = df_base.copy()
    if not df_gdelt.empty:
        df_signals = df_signals.merge(df_gdelt, on="date", how="left")
    if not df_bce_agg.empty:
        df_signals = df_signals.merge(df_bce_agg, on="date", how="left")
    if not df_ine_agg.empty:
        df_signals = df_signals.merge(df_ine_agg, on="date", how="left")

    # Fill NaN with defaults
    fill = {
        "gdelt_avg_tone": 0.0,
        "gdelt_goldstein": 0.0,
        "gdelt_n_articles": 0,
        "bce_shock_score": 0.0,
        "bce_uncertainty": 0.5,
        "bce_tone": "neutral",
        "ine_surprise_score": 0.0,
        "ine_topic": "otro",
        # derived
        "signal_available": 0.0,
        "bce_tone_numeric": 0.0,
        "bce_cumstance": 0.0,
        "gdelt_tone_ma3": 0.0,
        "gdelt_tone_ma6": 0.0,
        "ine_inflacion": 0.0,
    }
    for col_name, val in fill.items():
        if col_name not in df_signals.columns:
            df_signals[col_name] = val
        else:
            df_signals[col_name] = df_signals[col_name].fillna(val)

    df_signals["gdelt_n_articles"] = df_signals["gdelt_n_articles"].astype(int)

    # dominant_topic: captures stable monetary policy months where BCE shock > 0
    def _dominant(row):
        if row["bce_tone"] in ("hawkish", "dovish") or row["bce_shock_score"] > 0:
            return "tipos_interes"
        if row["ine_topic"] != "otro":
            return row["ine_topic"]
        return "otro"

    df_signals["dominant_topic"] = df_signals.apply(_dominant, axis=1)

    # Derived features — sort by date before computing rolling windows
    df_signals = df_signals.sort_values("date").reset_index(drop=True)

    # 1. Signal availability indicator (0 before 2015, 1 from 2015)
    #    Lets models distinguish "real zeros" from "no historical data"
    df_signals["signal_available"] = (df_signals["date"] >= "2015-01").astype(float)

    # 2. BCE tone as ordinal numeric: hawkish=1, neutral=0, dovish=-1
    _tone_map = {"hawkish": 1.0, "neutral": 0.0, "dovish": -1.0}
    df_signals["bce_tone_numeric"] = df_signals["bce_tone"].map(_tone_map).fillna(0.0)

    # 3. BCE cumulative stance: running sum of tone_numeric
    #    Captures tightening (positive) or easing (negative) cycles
    _stance = df_signals["bce_tone_numeric"].copy()
    _stance[df_signals["signal_available"] == 0] = 0.0
    df_signals["bce_cumstance"] = _stance.cumsum()

    # 4. GDELT tone moving averages (smooth monthly noise)
    #    min_periods=1 avoids NaN in early months of the range
    df_signals["gdelt_tone_ma3"] = df_signals["gdelt_avg_tone"].rolling(3, min_periods=1).mean()
    df_signals["gdelt_tone_ma6"] = df_signals["gdelt_avg_tone"].rolling(6, min_periods=1).mean()

    # 5. INE topic as binary (1=inflacion, 0=other)
    df_signals["ine_inflacion"] = (df_signals["ine_topic"] == "inflacion").astype(float)

    df_signals["date"] = pd.to_datetime(df_signals["date"] + "-01")

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df_signals.to_parquet(SIGNALS_PATH, index=False)
    logger.info(f"[build-c1] news_signals.parquet: {len(df_signals)} rows")
    logger.info(f"           Columns: {list(df_signals.columns)}")

    # Merge with features_exog to build features_c1
    if not FEATURES_EXOG_PATH.exists():
        logger.warning("[build-c1] features_exog.parquet not found. Only news_signals generated.")
        return

    df_exog = pd.read_parquet(FEATURES_EXOG_PATH)
    if "date" not in df_exog.columns:
        df_exog = df_exog.reset_index()
        df_exog = df_exog.rename(columns={df_exog.columns[0]: "date"})
    df_exog["date"] = pd.to_datetime(df_exog["date"])

    # Leakage fix: shift signals +1 month
    # Signals from month t are not available until after month t ends.
    # Shifting moves signal[t] to row[t+1], so features_c1[t] contains signals[t-1].
    signal_cols = [c for c in df_signals.columns if c != "date"]
    df_signals_lagged = df_signals.copy()
    df_signals_lagged[signal_cols] = df_signals_lagged[signal_cols].shift(1)
    for col_name, val in fill.items():
        if col_name in df_signals_lagged.columns:
            df_signals_lagged[col_name] = df_signals_lagged[col_name].fillna(val)
    df_signals_lagged["dominant_topic"] = df_signals_lagged["dominant_topic"].fillna("otro")
    df_signals_lagged["bce_tone"] = df_signals_lagged["bce_tone"].fillna("neutral")
    df_signals_lagged["ine_topic"] = df_signals_lagged["ine_topic"].fillna("otro")

    df_c1 = df_exog.merge(
        df_signals_lagged[["date"] + signal_cols],
        on="date",
        how="left",
    )

    for col_name, val in fill.items():
        if col_name in df_c1.columns:
            df_c1[col_name] = df_c1[col_name].fillna(val)
    if "dominant_topic" in df_c1.columns:
        df_c1["dominant_topic"] = df_c1["dominant_topic"].fillna("otro")

    df_c1.to_parquet(FEATURES_C1_PATH, index=False)
    logger.info(f"[build-c1] features_c1.parquet: {len(df_c1)} rows, {len(df_c1.columns)} cols")
    logger.info(f"           [leakage fix] signals shifted +1 month (signals[t] in row[t+1])")
    logger.info(f"           Columns: {list(df_c1.columns)}")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="MCP pipeline: acquire, LLM process, and build C1 features"
    )
    parser.add_argument(
        "--acquire", action="store_true",
        help="Download GDELT + RSS and store in MongoDB (requires Internet)",
    )
    parser.add_argument(
        "--process", action="store_true",
        help="Process pending RSS releases with Qwen3:4b (requires Ollama)",
    )
    parser.add_argument(
        "--build-c1", action="store_true",
        help="Aggregate monthly signals and export parquet",
    )
    parser.add_argument(
        "--start", type=str, default=GDELT_START,
        help=f"Start date (default: {GDELT_START})",
    )
    parser.add_argument(
        "--end", type=str, default=GDELT_END,
        help=f"End date (default: {GDELT_END})",
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
