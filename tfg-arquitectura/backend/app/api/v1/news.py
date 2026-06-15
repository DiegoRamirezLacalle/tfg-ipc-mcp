"""Live news pulse - GDELT ingestion + FinBERT sentiment via MCP.

Public (no auth) read endpoints so the landing page can show a live pulse:

  POST /api/v1/news/refresh        - ingest latest GDELT articles into MongoDB
  GET  /api/v1/news/today          - latest cached articles + meta (fast, from Mongo)
  GET  /api/v1/news/sentiment      - FinBERT aggregate for a month (slow, via MCP)

GDELT is strictly rate-limited ("one request every 5 seconds"), so /refresh is
guarded server-side and the page always reads cached articles from Mongo.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Query

from app.db.mongo import get_mongo_db
from app.mcp.client import fetch_news_sentiment
from app.services.gdelt import GdeltRateLimited, fetch_gdelt_articles

router = APIRouter(prefix="/news", tags=["news"])
log = structlog.get_logger()

_REFRESH_MIN_INTERVAL_S = 8  # respect GDELT's 1-req/5s, with margin
_LATEST_LIMIT = 12


async def _get_meta(db) -> dict:
    return await db["news_meta"].find_one({"_id": "gdelt"}) or {}


@router.post("/refresh")
async def refresh_news(
    timespan: str = Query("3d", description="GDELT lookback window, e.g. 1d/3d/7d"),
) -> dict:
    """Ingest the latest GDELT articles into news_raw (rate-guarded)."""
    db = get_mongo_db()
    meta = await _get_meta(db)

    now = datetime.now(timezone.utc)
    last_iso = meta.get("last_refresh")
    if last_iso:
        try:
            elapsed = (now - datetime.fromisoformat(last_iso)).total_seconds()
            if elapsed < _REFRESH_MIN_INTERVAL_S:
                total = await db["news_raw"].count_documents({})
                return {
                    "ingested": 0,
                    "total": total,
                    "skipped": True,
                    "reason": f"refreshed {int(elapsed)}s ago - try again shortly",
                    "last_refresh": last_iso,
                }
        except ValueError:
            pass

    try:
        articles = await fetch_gdelt_articles(timespan=timespan)
    except GdeltRateLimited:
        total = await db["news_raw"].count_documents({})
        return {
            "ingested": 0,
            "total": total,
            "rate_limited": True,
            "reason": "GDELT rate limit hit - showing cached articles",
            "last_refresh": last_iso,
        }
    except Exception as exc:  # noqa: BLE001
        total = await db["news_raw"].count_documents({})
        log.warning("news_refresh_failed", error=str(exc))
        return {
            "ingested": 0,
            "total": total,
            "error": str(exc)[:160],
            "last_refresh": last_iso,
        }

    ingested = 0
    for art in articles:
        res = await db["news_raw"].update_one(
            {"url": art["url"]},
            {"$set": art, "$setOnInsert": {"fetched_at": now.isoformat()}},
            upsert=True,
        )
        if res.upserted_id is not None:
            ingested += 1

    await db["news_meta"].update_one(
        {"_id": "gdelt"},
        {"$set": {"last_refresh": now.isoformat(), "last_fetch_count": len(articles)}},
        upsert=True,
    )
    total = await db["news_raw"].count_documents({})
    log.info("news_refreshed", ingested=ingested, fetched=len(articles), total=total)
    return {"ingested": ingested, "fetched": len(articles), "total": total,
            "last_refresh": now.isoformat()}


@router.get("/today")
async def news_today(limit: int = Query(_LATEST_LIMIT, ge=1, le=50)) -> dict:
    """Return the most recent cached articles plus pulse metadata."""
    db = get_mongo_db()
    cursor = (
        db["news_raw"]
        .find({}, {"_id": 0, "title": 1, "date": 1, "source": 1, "url": 1, "language": 1})
        .sort("date", -1)
        .limit(limit)
    )
    articles = await cursor.to_list(length=limit)
    total = await db["news_raw"].count_documents({})
    meta = await _get_meta(db)
    latest_month = articles[0]["date"][:7] if articles else None

    return {
        "articles": articles,
        "total": total,
        "latest_month": latest_month,
        "last_refresh": meta.get("last_refresh"),
    }


@router.get("/sentiment")
async def news_sentiment(
    year_month: str = Query(..., description="Month to score, YYYY-MM"),
    country: str = Query("global", description="Country filter for the FinBERT scorer"),
) -> dict:
    """FinBERT sentiment aggregate for a month, computed by the MCP server."""
    result = await fetch_news_sentiment(country=country, year_month=year_month)
    if result is None:
        return {
            "year_month": year_month,
            "available": False,
            "message": "Sentiment unavailable (MCP/FinBERT not reachable or no articles).",
        }
    return {"available": True, **result}
