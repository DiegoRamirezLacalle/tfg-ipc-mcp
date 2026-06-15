"""Ollama narration service.

Calls the local Ollama HTTP API (running on the Docker host) to generate
a concise natural-language analysis of a completed forecast run.

The model (llama3.2:3b by default) is configurable via OLLAMA_MODEL in .env.
When Ollama is unavailable the caller receives an httpx error which the
endpoint converts to 503.
"""

from __future__ import annotations

import httpx
import structlog

from app.config import settings

log = structlog.get_logger()


async def generate_narration(
    model_slug: str,
    metrics: dict[str, float],
    predictions: list[float],
    use_mcp: bool = False,
    mcp_signals: list[dict] | None = None,
) -> str:
    """Call Ollama and return the generated narrative string."""
    prompt = _build_prompt(model_slug, metrics, predictions, use_mcp, mcp_signals)

    url = f"{settings.OLLAMA_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 350},
    }

    log.info("narration_request", model=settings.OLLAMA_MODEL, url=url)
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        log.info("narration_done", chars=len(text))
        return text


def _build_prompt(
    model_slug: str,
    metrics: dict[str, float],
    predictions: list[float],
    use_mcp: bool,
    mcp_signals: list[dict] | None,
) -> str:
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    mape = metrics.get("mape")

    pred_str = ", ".join(f"{p:.2f}" for p in predictions[:6])
    if len(predictions) > 6:
        pred_str += f", ... ({len(predictions)} steps total)"

    mcp_note = ""
    if use_mcp and mcp_signals:
        first = next(
            (s for s in mcp_signals if s.get("available", True) and "error" not in s),
            None,
        )
        if first:
            ecb = first.get("ecb_hawkish_score")
            fomc = first.get("fomc_hawkish_score")
            parts = []
            if ecb is not None:
                parts.append(f"ECB hawkishness {ecb:.2f}")
            if fomc is not None:
                parts.append(f"FOMC hawkishness {fomc:.2f}")
            if parts:
                mcp_note = " | " + ", ".join(parts) + " (0=dovish, 1=hawkish)"

    mae_str  = f"{mae:.4f}"  if mae  is not None else "N/A"
    rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
    mape_str = f"{mape:.2f}%" if mape is not None else "N/A"

    return f"""You are a concise inflation analyst writing for a research report.

Forecast summary:
- Model: {model_slug}
- MCP macro context: {"yes" + mcp_note if use_mcp else "no"}
- Predicted YoY inflation (%): [{pred_str}]
- MAE: {mae_str}  RMSE: {rmse_str}  MAPE: {mape_str}

Write a professional 3-4 sentence analysis of the forecast trajectory and model accuracy. \
Reference the specific numbers. Comment on whether the model is performing well for inflation \
forecasting and any notable trend in the predictions. Do not use bullet points or headers."""
