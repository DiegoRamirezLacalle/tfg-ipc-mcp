"""GDELT DOC 2.0 news ingestion.

Fetches recent inflation-related news headlines from the free GDELT DOC API
(no API key required) and normalises them into the `news_raw` schema the MCP
server already understands ({title, body, date, source}).

GDELT enforces a strict rate limit ("one request every 5 seconds"); callers
must rate-guard. We surface 429 as GdeltRateLimited so the API can degrade
gracefully to whatever is already cached in MongoDB.
"""

from __future__ import annotations

import httpx
import structlog

log = structlog.get_logger()

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# Broad-but-focused inflation query (English + Spanish coverage via GDELT translation).
DEFAULT_QUERY = '(inflation OR "consumer prices" OR "central bank")'
_UA = "tfg-ipc-mcp/1.0 (research; inflation forecasting platform)"


class GdeltRateLimited(Exception):
    """Raised when GDELT returns HTTP 429 (too many requests)."""


def _to_date(seendate: str) -> str | None:
    """GDELT seendate '20260524T120000Z' -> '2026-05-24'."""
    if len(seendate) >= 8 and seendate[:8].isdigit():
        return f"{seendate[0:4]}-{seendate[4:6]}-{seendate[6:8]}"
    return None


async def fetch_gdelt_articles(
    query: str = DEFAULT_QUERY,
    timespan: str = "3d",
    maxrecords: int = 50,
) -> list[dict]:
    """Fetch and normalise recent GDELT articles.

    Returns a list of dicts ready to upsert into news_raw. Raises
    GdeltRateLimited on 429, httpx errors on transport failure.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "timespan": timespan,
        "maxrecords": str(maxrecords),
    }
    async with httpx.AsyncClient(timeout=45.0, headers={"User-Agent": _UA}) as client:
        resp = await client.get(GDELT_URL, params=params)

    if resp.status_code == 429:
        log.warning("gdelt_rate_limited")
        raise GdeltRateLimited(resp.text[:200])
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as exc:  # GDELT occasionally returns non-JSON on edge cases
        log.warning("gdelt_bad_json", error=str(exc), head=resp.text[:120])
        return []

    out: list[dict] = []
    seen_urls: set[str] = set()
    for art in data.get("articles", []):
        url = (art.get("url") or "").strip()
        title = (art.get("title") or "").strip()
        date = _to_date(art.get("seendate", ""))
        if not url or not title or not date or url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(
            {
                "url": url,
                "title": title,
                "body": title,  # GDELT DOC gives no article body; title scores fine
                "date": date,
                "source": (art.get("domain") or "").strip(),
                "language": (art.get("language") or "").strip(),
                "seendate": art.get("seendate", ""),
                "image": (art.get("socialimage") or "").strip(),
            }
        )
    log.info("gdelt_fetched", n=len(out), query=query, timespan=timespan)
    return out
