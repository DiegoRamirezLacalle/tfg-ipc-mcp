"""Simulator chat assistant — streaming LLM tutor.

POST /api/v1/assistant/simulator/chat

Multi-turn chat backed by the same local Ollama instance that powers the run
narration (model from settings.OLLAMA_MODEL). The system prompt teaches the
simulator's mechanics AND the thesis context, and the *live* simulator state
(series, current slider values, deltas, top driver) is injected per request so
the bot can answer "why did *my* forecast change?" with the actual numbers.

The response streams token-by-token (text/plain) so the UI feels responsive on
small models like llama3.2:3b.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Literal

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.core.permissions import get_current_user
from app.models.user import User

router = APIRouter(prefix="/assistant", tags=["assistant"])
log = structlog.get_logger()

_MAX_MESSAGES = 30
_MAX_USER_CHARS = 1500
_STREAM_TIMEOUT_S = 120


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=_MAX_USER_CHARS)


class SignalState(BaseModel):
    key: str
    label: str
    baseline_value: float
    current_value: float
    final_effect: float | None = None  # effect at the last horizon step


class SimChatContext(BaseModel):
    series_name: str | None = None
    series_unit: str | None = None
    horizon: int = 6
    signals: list[SignalState] = []
    baseline: list[float] = []
    counterfactual: list[float] = []
    top_driver_label: str | None = None
    top_driver_contribution: float | None = None


class SimChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: SimChatContext | None = None


_SYSTEM_INTRO = """You are the in-app tutor for an inflation-forecasting research \
platform built as a Bachelor's thesis (TFG).

Concepts you teach in plain language:

- This page is the "What-if Simulator". It perturbs macro / monetary-policy signals \
(ECB Deposit Rate, Fed/FOMC stance, Fed forward guidance, US CPI direction) and \
shows how the forecast responds.
- The Counterfactual is computed as: counterfactual[d] = baseline[d] + \
sum_i (slider_i - baseline_value_i) * effect_i[d], where effect_i[d] is a \
per-horizon Ridge marginal effect.
- The Ridge fit uses the h-step *change* of the target (not the level), so the \
effect isolates the genuine predictive contribution and avoids the spurious \
level correlation the thesis warns about.
- Baseline = an ARIMA-based time-series forecast that ignores signals. \
Counterfactual = baseline + signal-driven correction.
- The thesis research questions: RQ1 — Do foundation models (TimesFM, Chronos-2, \
TimeGPT) beat classical baselines (ARIMA, SARIMA)? RQ2 — Do MCP exogenous signals \
add predictive value?
- The MCP (Model Context Protocol) is what fetches the macro signals and the \
FinBERT news sentiment that the forecasters consume as C1 context.

Style:
- Keep answers short (2-5 sentences) and concrete. Use the live numbers from the \
context when relevant; quote specific values and horizons.
- Speak English with plain financial vocabulary.
- If the user asks something unrelated to forecasting / inflation / this platform, \
politely redirect.
- Never invent numbers that aren't in the live context. If the context lacks the \
detail, say so.
- Reply directly without any visible chain-of-thought. /no_think"""


def _format_context(ctx: SimChatContext | None) -> str:
    if ctx is None:
        return "Live simulator state: not provided."

    lines = ["Live simulator state:"]
    if ctx.series_name:
        unit = f" ({ctx.series_unit})" if ctx.series_unit else ""
        lines.append(f"- Target series: {ctx.series_name}{unit}")
    lines.append(f"- Forecast horizon: {ctx.horizon} months")

    if ctx.signals:
        lines.append("- Signal levers:")
        for s in ctx.signals:
            delta = s.current_value - s.baseline_value
            eff = ""
            if s.final_effect is not None:
                eff = f" · per-unit effect at last h ≈ {s.final_effect:+.3f}"
            lines.append(
                f"    · {s.label} (key={s.key}): baseline={s.baseline_value:.3f}, "
                f"current={s.current_value:.3f}, Δ={delta:+.3f}{eff}"
            )

    if ctx.baseline and ctx.counterfactual:
        deltas = [c - b for b, c in zip(ctx.baseline, ctx.counterfactual)]
        if deltas:
            lines.append(
                f"- Forecast deltas (counterfactual − baseline): "
                f"h1={deltas[0]:+.3f}, h{len(deltas)}={deltas[-1]:+.3f}, "
                f"max|Δ|={max(abs(d) for d in deltas):.3f}"
            )
    if ctx.top_driver_label and ctx.top_driver_contribution is not None:
        lines.append(
            f"- Top driver of the current Δ at the final horizon: "
            f"{ctx.top_driver_label} (contribution {ctx.top_driver_contribution:+.3f})"
        )
    return "\n".join(lines)


def _build_messages(req: SimChatRequest) -> list[dict]:
    system = _SYSTEM_INTRO + "\n\n" + _format_context(req.context)
    msgs = [{"role": "system", "content": system}]
    # Keep only the last N exchanges
    for m in req.messages[-_MAX_MESSAGES:]:
        msgs.append({"role": m.role, "content": m.content})
    return msgs


async def _stream_ollama(messages: list[dict]) -> AsyncIterator[bytes]:
    url = f"{settings.OLLAMA_URL}/api/chat"
    payload = {
        "model": settings.OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": True,
        # Disable chain-of-thought on thinking models (Qwen 3) - we want fast,
        # streamed tokens, not minutes of hidden reasoning. The system prompt
        # also carries the /no_think directive as a fallback for older runtimes.
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 400},
    }
    try:
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT_S) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    detail = body.decode("utf-8", "ignore")[:200]
                    yield f"[Ollama error {resp.status_code}: {detail}]".encode()
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    piece = msg.get("content", "")
                    if piece:
                        yield piece.encode("utf-8")
                    if chunk.get("done"):
                        break
    except httpx.ConnectError:
        yield (
            b"[Ollama unreachable. Make sure Ollama is running "
            b"on the host (host.docker.internal:11434).]"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("assistant_stream_failed", error=str(exc))
        yield f"[Assistant error: {str(exc)[:200]}]".encode()


@router.post("/simulator/chat")
async def simulator_chat(
    req: SimChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream a chat response token-by-token. Body is plain UTF-8 text."""
    if not req.messages:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "messages must be non-empty")
    if req.messages[-1].role != "user":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "last message must be from user")

    messages = _build_messages(req)
    log.info(
        "assistant_chat",
        user_id=current_user.id,
        n_messages=len(req.messages),
        has_context=req.context is not None,
        n_signals=len(req.context.signals) if req.context else 0,
    )

    return StreamingResponse(_stream_ollama(messages), media_type="text/plain; charset=utf-8")
