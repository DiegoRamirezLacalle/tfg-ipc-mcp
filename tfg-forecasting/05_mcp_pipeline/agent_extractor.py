"""
agent_extractor.py
------------------
Procesa textos RSS de comunicados oficiales (BCE, INE, BdE) con Ollama
(qwen3:4b) y devuelve senales estructuradas via Pydantic + structured output.

GDELT NO pasa por aqui — sus senales son puramente cuantitativas.

Requiere:
    1. Ollama instalado y corriendo
    2. pip install ollama pydantic
    3. ollama pull qwen3:4b
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import ollama
from pydantic import BaseModel, Field

# ── Configuracion ──────────────────────────────────────────────
MODEL = "qwen3:4b"
PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction_v1.txt"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── Esquema Pydantic para salida estructurada ──────────────────
class Decision(str, Enum):
    subida = "subida"
    bajada = "bajada"
    sin_cambio = "sin_cambio"
    dato = "dato"


class Tone(str, Enum):
    hawkish = "hawkish"
    neutral = "neutral"
    dovish = "dovish"
    positivo = "positivo"
    negativo = "negativo"


class Topic(str, Enum):
    tipos_interes = "tipos_interes"
    inflacion = "inflacion"
    empleo = "empleo"
    pib = "pib"
    otro = "otro"


class RSSSignals(BaseModel):
    """Senales extraidas de un comunicado oficial."""

    decision: Decision = Field(
        description="subida|bajada|sin_cambio|dato",
    )
    magnitude: Optional[float] = Field(
        default=None,
        description="Numeric change (e.g. 0.50 for 50bps). null if not applicable.",
    )
    tone: Tone = Field(
        description="hawkish|neutral|dovish|positivo|negativo",
    )
    shock_score: float = Field(
        ge=0.0, le=1.0,
        description="0=expected, 1=completely unexpected",
    )
    uncertainty_index: float = Field(
        ge=0.0, le=1.0,
        description="0=clear outlook, 1=maximum ambiguity",
    )
    topic: Topic = Field(
        description="tipos_interes|inflacion|empleo|pib|otro",
    )


# Valores por defecto para texto irrelevante o error
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
    """Elimina bloques <think>...</think> de Qwen3."""
    return _THINK_RE.sub("", text).strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def extract_signals(text: str, source: str = "") -> dict[str, Any]:
    """
    Extrae senales de un comunicado oficial.

    Parameters
    ----------
    text : str
        Titulo + cuerpo del comunicado RSS.
    source : str
        Fuente: "bce", "ine", "bde" (para contexto).

    Returns
    -------
    dict con decision, magnitude, tone, shock_score, uncertainty_index, topic.
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
            think=False,  # desactiva thinking mode de Qwen3 (evita <think> largos)
            options={"temperature": 0},
        )
        raw_text = _strip_thinking(response["message"]["content"])
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "pull" in err_msg:
            raise RuntimeError(
                f"Modelo '{MODEL}' no descargado. Ejecuta:\n  ollama pull {MODEL}"
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
        print(f"[WARN] Extraccion fallida: {exc}")
        print(f"       Respuesta LLM: {raw_text[:300]}")
        return DEFAULT_SIGNALS.model_dump()


# ── CLI para test manual ──────────────────────────────────────
if __name__ == "__main__":
    test_text = (
        "BCE sube tipos 50 puntos basicos. "
        "El Consejo de Gobierno del Banco Central Europeo ha decidido "
        "subir los tres tipos de interes oficiales del BCE en 50 puntos "
        "basicos. La inflacion sigue siendo demasiado alta y se preve "
        "que se mantenga por encima del objetivo durante un periodo "
        "prolongado."
    )
    print(f"Modelo: {MODEL}")
    print(f"Schema: {json.dumps(RSSSignals.model_json_schema(), indent=2)}")
    print("\nExtrayendo senales...")
    result = extract_signals(test_text, source="bce")
    print(json.dumps(result, indent=2))
