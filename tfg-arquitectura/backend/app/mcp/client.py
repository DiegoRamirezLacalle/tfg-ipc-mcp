"""Async MCP client for the platform backend.

Connects to the MCP server (SSE transport) and fetches:
  - macro context signals (get_macro_signals) per forecast timestamp
  - GDELT news sentiment (get_news_sentiment) per forecast timestamp

Returns empty list on any failure so forecast runs are never blocked
by MCP unavailability.
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from app.config import settings

log = structlog.get_logger()


def _mcp_url() -> str:
    return settings.MCP_SERVER_URL


async def fetch_signals_for_timestamps(
    timestamps: list,
    country: str = "spain",
) -> list[dict]:
    """Call get_macro_signals + get_news_sentiment for each forecast timestamp.

    Args:
        timestamps: list of pd.Timestamp / datetime objects (forecast steps).
        country:    Country for news sentiment scoring (default "spain").

    Returns:
        List of signal dicts, one per timestamp. Returns [] on any error.
        Each dict contains macro signals + sentiment_{mean,std,n_articles,hawkish}.
    """
    try:
        from mcp import ClientSession  # noqa: PLC0415
        from mcp.client.sse import sse_client  # noqa: PLC0415
    except ImportError:
        log.warning("mcp_client_unavailable", reason="mcp package not installed")
        return []

    results: list[dict] = []
    try:
        async with sse_client(_mcp_url()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                for ts in timestamps:
                    ym = ts.strftime("%Y-%m") if isinstance(ts, datetime) else str(ts)[:7]
                    row: dict = {"year_month": ym}

                    # ── macro signals ──────────────────────────────────────────
                    try:
                        raw = await session.call_tool(
                            "get_macro_signals", {"year_month": ym}
                        )
                        payload = json.loads(raw.content[0].text)
                        if payload.get("available"):
                            row.update(payload.get("signals", {}))
                        else:
                            row["macro_available"] = False
                    except Exception as tool_err:
                        log.warning("mcp_macro_error", year_month=ym, error=str(tool_err))
                        row["macro_error"] = str(tool_err)

                    # ── news sentiment (FinBERT) ───────────────────────────────
                    try:
                        raw_sent = await session.call_tool(
                            "get_news_sentiment",
                            {"country": country, "year_month": ym},
                        )
                        sent = json.loads(raw_sent.content[0].text)
                        if "error" not in sent:
                            row["sentiment_mean"]     = sent.get("sentiment_mean")
                            row["sentiment_std"]      = sent.get("sentiment_std")
                            row["sentiment_n"]        = sent.get("n_articles")
                            row["sentiment_hawkish"]  = sent.get("hawkish_score")
                    except Exception as sent_err:
                        log.warning("mcp_sentiment_error", year_month=ym, error=str(sent_err))

                    results.append(row)

    except Exception as conn_err:
        log.warning(
            "mcp_server_unreachable",
            url=_mcp_url(),
            error=str(conn_err),
        )
        return []

    return results


async def fetch_news_sentiment(country: str, year_month: str) -> dict | None:
    """Call the MCP get_news_sentiment tool (FinBERT) for one country/month.

    Returns the parsed sentiment dict, or None if the MCP server is unreachable
    or the tool errors. FinBERT lazy-loads on first call, so the first request
    may take 30-60s while the model downloads/initialises.
    """
    try:
        from mcp import ClientSession  # noqa: PLC0415
        from mcp.client.sse import sse_client  # noqa: PLC0415
    except ImportError:
        log.warning("mcp_client_unavailable", reason="mcp package not installed")
        return None

    try:
        async with sse_client(_mcp_url()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                raw = await session.call_tool(
                    "get_news_sentiment",
                    {"country": country, "year_month": year_month},
                )
                payload = json.loads(raw.content[0].text)
                if "error" in payload:
                    log.warning("mcp_sentiment_tool_error", error=payload.get("error"))
                    return None
                return payload
    except Exception as conn_err:
        log.warning("mcp_sentiment_unreachable", url=_mcp_url(), error=str(conn_err))
        return None
