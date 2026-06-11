"""MCP server for the TFG platform (SSE HTTP transport).

Exposes 3 tools to the platform backend:

  1. get_macro_signals(year_month)
       Returns pre-computed ECB / FOMC / CPI signals for a calendar month.
       Source: tfg-forecasting/data/processed/mcp_signals_global.parquet

  2. get_news_context(start_date, end_date, topic)
       Searches MongoDB news_raw for articles matching topic + date range.
       Returns the 10 most recent articles with title, body, date, source.

  3. get_news_sentiment(country, year_month)
       Scores GDELT news articles for a country/month using FinBERT.
       Returns {n_articles, sentiment_mean, sentiment_std, hawkish_score}.

The server runs on port MCP_SERVER_PORT (default 8080).
"""

from __future__ import annotations

import calendar
import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient

# -- paths ---------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "tfg-forecasting" / "data" / "processed"
_SIGNALS_PATH = _DATA_DIR / "mcp_signals_global.parquet"

# -- lazy parquet load ---------------------------------------------------------

_signals_df: pd.DataFrame | None = None


def _get_signals() -> pd.DataFrame:
    global _signals_df
    if _signals_df is None:
        _signals_df = pd.read_parquet(_SIGNALS_PATH)
        _signals_df.index = pd.to_datetime(_signals_df.index)
    return _signals_df


# -- FinBERT lazy load (double-checked locking) --------------------------------

_FINBERT_MODEL = "ProsusAI/finbert"
_FINBERT_LOCK = threading.Lock()
_FINBERT_PIPELINE = None


def _get_sentiment_pipeline():
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        with _FINBERT_LOCK:
            if _FINBERT_PIPELINE is None:
                from transformers import pipeline  # noqa: PLC0415
                _FINBERT_PIPELINE = pipeline(
                    "sentiment-analysis",
                    model=_FINBERT_MODEL,
                    truncation=True,
                    max_length=512,
                    device=-1,  # CPU only
                )
    return _FINBERT_PIPELINE


# Keywords that signal hawkish monetary policy stance
_HAWKISH_KEYWORDS = frozenset({
    "rate hike", "rate increase", "tightening", "hawkish", "raise rates",
    "subir tipos", "subida de tipos", "restricción monetaria", "política restrictiva",
    "inflation fight", "combat inflation", "lucha contra la inflación",
    "bajar inflación", "reducir inflación", "endurecimiento monetario",
    "interest rate rise", "monetary tightening", "hike rates",
})

# -- MongoDB helper ------------------------------------------------------------


def _news_collection():
    uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    return client["tfg_news"]["news_raw"]


# -- FastMCP server ------------------------------------------------------------

_port = int(os.getenv("MCP_SERVER_PORT", "8080"))
mcp = FastMCP(name="tfg-ipc-macro-context", host="0.0.0.0", port=_port)


@mcp.tool()
def get_macro_signals(year_month: str) -> str:
    """Return pre-computed ECB / FOMC / CPI macro signals for a calendar month.

    Args:
        year_month: Month in YYYY-MM format (e.g. "2024-03").

    Returns:
        JSON object with signal values, or {"error": "..."} if not available.
    """
    try:
        ts = pd.Timestamp(year_month + "-01")
    except Exception:
        return json.dumps({"error": f"Invalid year_month format: {year_month!r}. Use YYYY-MM."})

    df = _get_signals()
    if ts not in df.index:
        return json.dumps({"year_month": year_month, "available": False, "signals": {}})

    row = df.loc[ts].to_dict()
    return json.dumps({
        "year_month": year_month,
        "available": True,
        "signals": {k: round(float(v), 6) if v == v else None for k, v in row.items()},
    })


@mcp.tool()
def get_news_context(start_date: str, end_date: str, topic: str = "inflacion") -> str:
    """Search MongoDB news_raw for macro articles matching a topic and date range.

    Args:
        start_date: Start date, YYYY-MM-DD format.
        end_date:   End date, YYYY-MM-DD format.
        topic:      Keyword topic to search (default "inflacion").

    Returns:
        JSON array of up to 10 articles: [{title, body, date, source}, ...].
        Returns [] if MongoDB is unavailable.
    """
    try:
        col = _news_collection()
        docs = list(
            col.find(
                {
                    "date": {"$gte": start_date, "$lte": end_date},
                    "$or": [
                        {"title": {"$regex": topic, "$options": "i"}},
                        {"body": {"$regex": topic, "$options": "i"}},
                    ],
                },
                {"_id": 0, "title": 1, "body": 1, "date": 1, "source": 1},
            )
            .sort("date", -1)
            .limit(10)
        )
        return json.dumps(docs, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "articles": []})


@mcp.tool()
def get_news_sentiment(country: str, year_month: str) -> str:
    """Score GDELT news articles for a country/month using FinBERT sentiment analysis.

    Fetches up to 50 articles from MongoDB news_raw, scores each title+body
    with ProsusAI/finbert, and aggregates to monthly sentiment statistics.

    Args:
        country:    Country name or ISO code (e.g. "spain", "ES", "global").
        year_month: Month in YYYY-MM format.

    Returns:
        JSON with {n_articles, sentiment_mean, sentiment_std, hawkish_score}.
        sentiment_mean: mean of (pos_prob - neg_prob), range [-1, +1].
        hawkish_score:  fraction of articles containing hawkish keywords.
    """
    try:
        year  = int(year_month[:4])
        month = int(year_month[5:7])
        start_date = f"{year_month}-01"
        end_date   = f"{year_month}-{calendar.monthrange(year, month)[1]:02d}"

        col = _news_collection()
        query: dict = {"date": {"$gte": start_date, "$lte": end_date}}

        country_lower = country.lower().strip()
        if country_lower not in ("", "global", "world"):
            kw = (
                "spain|españa|español|ipc|ine"
                if country_lower in ("spain", "es")
                else country_lower
            )
            query["$or"] = [
                {"title":  {"$regex": kw, "$options": "i"}},
                {"body":   {"$regex": kw, "$options": "i"}},
                {"source": {"$regex": country_lower, "$options": "i"}},
            ]

        docs = list(col.find(query, {"_id": 0, "title": 1, "body": 1}).limit(50))

        if not docs:
            return json.dumps({
                "year_month":     year_month,
                "country":        country,
                "n_articles":     0,
                "sentiment_mean": None,
                "sentiment_std":  None,
                "hawkish_score":  None,
            })

        nlp   = _get_sentiment_pipeline()
        texts = [
            (doc.get("title", "") + " " + doc.get("body", ""))[:512].strip()
            for doc in docs
        ]

        results = nlp(texts, batch_size=8)

        scores       = []
        hawkish_hits = 0
        for text, res in zip(texts, results):
            label = res["label"].lower()
            prob  = float(res["score"])
            scores.append(prob if label == "positive" else (-prob if label == "negative" else 0.0))

            if any(kw in text.lower() for kw in _HAWKISH_KEYWORDS):
                hawkish_hits += 1

        arr = np.array(scores, dtype=np.float64)

        return json.dumps({
            "year_month":     year_month,
            "country":        country,
            "n_articles":     len(docs),
            "sentiment_mean": round(float(np.mean(arr)), 4),
            "sentiment_std":  round(float(np.std(arr)),  4),
            "hawkish_score":  round(hawkish_hits / len(docs), 4),
        })

    except Exception as exc:
        return json.dumps({"error": str(exc), "year_month": year_month, "country": country})


# -- entry point ---------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="sse")
