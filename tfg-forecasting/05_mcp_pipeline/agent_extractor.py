"""Process RSS texts from official sources (ECB, INE, BdE) with Ollama
(qwen3:4b) and return structured signals via Pydantic + structured output.

GDELT is NOT processed here — its signals are purely quantitative.

Requirements:
    1. Ollama installed and running
    2. pip install ollama pydantic
    3. ollama pull qwen3:4b
"""

from __future__ import annotations

import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import ollama
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

# Configuration
MODEL = "qwen3:4b"
PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction_v1.txt"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# Pydantic schema for structured LLM output
class Decision(str, Enum):
    subida     = "subida"
    bajada     = "bajada"
    sin_cambio = "sin_cambio"
    dato       = "dato"


class Tone(str, Enum):
    hawkish  = "hawkish"
    neutral  = "neutral"
    dovish   = "dovish"
    positivo = "positivo"
    negativo = "negativo"


class Topic(str, Enum):
    tipos_interes = "tipos_interes"
    inflacion     = "inflacion"
    empleo        = "empleo"
    pib           = "pib"
    otro          = "otro"


class RSSSignals(BaseModel):
    """Signals extracted from an official press release."""

    decision: Decision = Field(description="subida|bajada|sin_cambio|dato")
    magnitude: Optional[float] = Field(
        default=None,
        description="Numeric change (e.g. 0.50 for 50bps). null if not applicable.",
    )
    tone: Tone = Field(description="hawkish|neutral|dovish|positivo|negativo")
    shock_score: float = Field(
        ge=0.0, le=1.0,
        description="0=expected, 1=completely unexpected",
    )
    uncertainty_index: float = Field(
        ge=0.0, le=1.0,
        description="0=clear outlook, 1=maximum ambiguity",
    )
    topic: Topic = Field(description="tipos_interes|inflacion|empleo|pib|otro")


# Default values for irrelevant text or parse error
DEFAULT_SIGNALS = RSSSignals(
    decision=Decision.dato,
    magnitude=None,
    tone=Tone.neutral,
    shock_score=0.0,
    uncertainty_index=0.5,
    topic=Topic.otro,
)


def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    return _THINK_RE.sub("", text).strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def extract_signals(text: str, source: str = "") -> dict[str, Any]:
    """Extract signals from an official press release.

    Parameters
    ----------
    text : str
        Title + body of the RSS entry.
    source : str
        Source identifier: "bce", "ine", or "bde" (used as context).

    Returns
    -------
    dict with decision, magnitude, tone, shock_score, uncertainty_index, topic.
    """
    if not text or not text.strip():
        return DEFAULT_SIGNALS.model_dump()

    user_content = f"Source: {source}\n\n{text[:800]}"

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": _load_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            format=RSSSignals.model_json_schema(),
            think=False,  # disable Qwen3 thinking mode (avoids long <think> blocks)
            options={"temperature": 0},
        )
        raw_text = _strip_thinking(response["message"]["content"])
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "pull" in err_msg:
            raise RuntimeError(
                f"Model '{MODEL}' not downloaded. Run:\n  ollama pull {MODEL}"
            ) from e
        raise

    try:
        raw = json.loads(raw_text)
        signals = RSSSignals(
            decision=raw.get("decision", "dato"),
            magnitude=raw.get("magnitude"),
            tone=raw.get("tone", "neutral"),
            shock_score=round(_clamp(float(raw.get("shock_score", 0)), 0.0, 1.0), 2),
            uncertainty_index=round(_clamp(float(raw.get("uncertainty_index", 0.5)), 0.0, 1.0), 2),
            topic=raw.get("topic", "otro"),
        )
        return signals.model_dump()
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning(f"Extraction failed: {exc}")
        logger.warning(f"LLM response: {raw_text[:300]}")
        return DEFAULT_SIGNALS.model_dump()


# CLI for manual testing
if __name__ == "__main__":
    test_text = (
        "BCE sube tipos 50 puntos basicos. "
        "El Consejo de Gobierno del Banco Central Europeo ha decidido "
        "subir los tres tipos de interes oficiales del BCE en 50 puntos "
        "basicos. La inflacion sigue siendo demasiado alta y se preve "
        "que se mantenga por encima del objetivo durante un periodo "
        "prolongado."
    )
    logger.info(f"Model: {MODEL}")
    logger.info(f"Schema: {json.dumps(RSSSignals.model_json_schema(), indent=2)}")
    logger.info("Extracting signals...")
    result = extract_signals(test_text, source="bce")
    logger.info(json.dumps(result, indent=2))
