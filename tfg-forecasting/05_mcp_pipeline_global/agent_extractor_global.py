"""
agent_extractor_global.py
--------------------------
Extrae señales LLM de comunicados globales (FOMC, ECB, BLS) con Qwen3:4b.

Esquemas Pydantic por fuente:
  FOMCSignals   → fomc_hawkish_score, fomc_surprise_score, fomc_forward_guidance
  ECBSignals    → ecb_hawkish_score, ecb_surprise_score, ecb_forward_guidance
  BLSSignals    → us_cpi_surprise_score, us_cpi_direction, us_cpi_components_pressure

Requiere:
    ollama pull qwen3:4b
    pip install ollama pydantic
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import ollama
from pydantic import BaseModel, Field

MODEL = "qwen3:4b"
PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction_global_v1.txt"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── Enums compartidos ─────────────────────────────────────────────────────

class ForwardGuidance(str, Enum):
    subida  = "subida"
    neutral = "neutral"
    bajada  = "bajada"

class RateDecision(str, Enum):
    subida    = "subida"
    bajada    = "bajada"
    sin_cambio = "sin_cambio"

class CPIDirection(str, Enum):
    aceleracion    = "aceleracion"
    estable        = "estable"
    desaceleracion = "desaceleracion"


# ── Schemas Pydantic ──────────────────────────────────────────────────────

class FOMCSignals(BaseModel):
    fomc_hawkish_score:   float = Field(ge=0.0, le=1.0)
    fomc_surprise_score:  float = Field(ge=0.0, le=1.0)
    fomc_forward_guidance: ForwardGuidance
    fomc_rate_decision:   RateDecision
    fomc_magnitude:       Optional[float] = None


class ECBSignals(BaseModel):
    ecb_hawkish_score:    float = Field(ge=0.0, le=1.0)
    ecb_surprise_score:   float = Field(ge=0.0, le=1.0)
    ecb_forward_guidance: ForwardGuidance
    ecb_rate_decision:    RateDecision
    ecb_magnitude:        Optional[float] = None


class BLSSignals(BaseModel):
    us_cpi_surprise_score:     float = Field(ge=0.0, le=1.0)
    us_cpi_direction:          CPIDirection
    us_cpi_components_pressure: float = Field(ge=0.0, le=1.0)


# Defaults para texto vacío o error
DEFAULTS = {
    "fomc": FOMCSignals(
        fomc_hawkish_score=0.5, fomc_surprise_score=0.0,
        fomc_forward_guidance="neutral", fomc_rate_decision="sin_cambio",
        fomc_magnitude=None,
    ).model_dump(),
    "ecb_press": ECBSignals(
        ecb_hawkish_score=0.5, ecb_surprise_score=0.0,
        ecb_forward_guidance="neutral", ecb_rate_decision="sin_cambio",
        ecb_magnitude=None,
    ).model_dump(),
    "bls_cpi": BLSSignals(
        us_cpi_surprise_score=0.0,
        us_cpi_direction="estable",
        us_cpi_components_pressure=0.5,
    ).model_dump(),
}

_SCHEMAS = {
    "fomc":     FOMCSignals,
    "ecb_press": ECBSignals,
    "bls_cpi":  BLSSignals,
}


# ── Utilidades ────────────────────────────────────────────────────────────

def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


# ── Función principal ─────────────────────────────────────────────────────

def extract_signals(text: str, source: str) -> dict[str, Any]:
    """
    Extrae señales LLM de un comunicado oficial.

    Parameters
    ----------
    text   : Título + cuerpo del comunicado (máx 1200 chars usados).
    source : "fomc" | "ecb_press" | "bls_cpi"

    Returns
    -------
    dict con las señales del esquema correspondiente.
    """
    if source not in _SCHEMAS:
        raise ValueError(f"Fuente no válida: {source}. Usar: {list(_SCHEMAS)}")

    if not text or not text.strip():
        return DEFAULTS[source]

    schema_cls = _SCHEMAS[source]

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": _load_prompt()},
                {"role": "user",   "content": f"Source: {source}\n\n{text[:1200]}"},
            ],
            format=schema_cls.model_json_schema(),
            think=False,
            options={"temperature": 0},
        )
        raw_text = _strip_thinking(response["message"]["content"])
    except Exception as e:
        err = str(e).lower()
        if "not found" in err or "pull" in err:
            raise RuntimeError(
                f"Modelo '{MODEL}' no instalado. Ejecuta:\n  ollama pull {MODEL}"
            ) from e
        print(f"[WARN] Ollama error ({source}): {e}")
        return DEFAULTS[source]

    try:
        raw = json.loads(raw_text)

        if source == "fomc":
            signals = FOMCSignals(
                fomc_hawkish_score=_clamp(raw.get("fomc_hawkish_score", 0.5)),
                fomc_surprise_score=_clamp(raw.get("fomc_surprise_score", 0.0)),
                fomc_forward_guidance=raw.get("fomc_forward_guidance", "neutral"),
                fomc_rate_decision=raw.get("fomc_rate_decision", "sin_cambio"),
                fomc_magnitude=raw.get("fomc_magnitude"),
            )
        elif source == "ecb_press":
            signals = ECBSignals(
                ecb_hawkish_score=_clamp(raw.get("ecb_hawkish_score", 0.5)),
                ecb_surprise_score=_clamp(raw.get("ecb_surprise_score", 0.0)),
                ecb_forward_guidance=raw.get("ecb_forward_guidance", "neutral"),
                ecb_rate_decision=raw.get("ecb_rate_decision", "sin_cambio"),
                ecb_magnitude=raw.get("ecb_magnitude"),
            )
        else:  # bls_cpi
            signals = BLSSignals(
                us_cpi_surprise_score=_clamp(raw.get("us_cpi_surprise_score", 0.0)),
                us_cpi_direction=raw.get("us_cpi_direction", "estable"),
                us_cpi_components_pressure=_clamp(raw.get("us_cpi_components_pressure", 0.5)),
            )

        return signals.model_dump()

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"[WARN] Extracción fallida ({source}): {exc}")
        print(f"       LLM: {raw_text[:200]}")
        return DEFAULTS[source]


# ── CLI test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = {
        "fomc": (
            "The Federal Open Market Committee decided to raise the target range for the "
            "federal funds rate to 5-1/4 to 5-1/2 percent. Inflation remains elevated. "
            "The Committee will continue reducing its holdings. Future increases may be appropriate."
        ),
        "ecb_press": (
            "The Governing Council decided to raise the three key ECB interest rates by 50 "
            "basis points. Inflation is far too high and is projected to remain above our 2% "
            "target for an extended period. Further rate increases are in store."
        ),
        "bls_cpi": (
            "The Consumer Price Index for All Urban Consumers increased 0.6 percent in September "
            "on a seasonally adjusted basis, much higher than expected. Over the last 12 months "
            "the all items index increased 8.2 percent. Shelter costs rose sharply, "
            "contributing significantly to broad-based price pressures."
        ),
    }

    for src, txt in tests.items():
        print(f"\n{'='*50}")
        print(f"Source: {src}")
        result = extract_signals(txt, src)
        print(json.dumps(result, indent=2))
