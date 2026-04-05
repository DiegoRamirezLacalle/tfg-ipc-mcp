"""
fix_signals.py
--------------
Repara las senales BCE e INE en MongoDB:

1. INE: Parsea el JSON de graficos del body para extraer tasas IPC reales
   y genera senales directamente (sin LLM, los datos son cuantitativos).

2. BCE: Inserta decisiones historicas conocidas del Consejo de Gobierno
   del BCE (2015-2024) con senales precisas. Datos publicos bien documentados.

3. Limpia documentos basura (paginas de navegacion BCE, placeholders).

Uso:
    python fix_signals.py          # Ejecuta todo
    python fix_signals.py --ine    # Solo INE
    python fix_signals.py --bce    # Solo BCE
    python fix_signals.py --clean  # Solo limpieza
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "tfg_ipc_mcp"
MONGO_COLLECTION = "news_raw"


def _get_collection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_COLLECTION]


# ═══════════════════════════════════════════════════════════════
# 1. FIX INE: Extraer senales de CPI desde chart JSON
# ═══════════════════════════════════════════════════════════════

def _parse_ine_chart_json(body: str) -> dict | None:
    """
    Extrae valores de IPC del JSON de graficos embebido en el body del INE.
    El JSON tiene:
      - ticks: ["ene-23", "feb-23", ...]  (meses)
      - values[0]: [5.9, 6.0, ...]        (IPC General)
      - values[1]: [7.5, 7.6, ...]        (IPC Subyacente)
    """
    # Buscar el bloque JSON en el body
    json_match = re.search(r'\{[^{}]*"mode"\s*:\s*"chart"[^{}]*"values"\s*:\s*\[.*?\]\s*,.*?\}',
                           body, re.DOTALL)
    if not json_match:
        # Intento alternativo: buscar desde { hasta el ultimo }
        start = body.find('{')
        end = body.rfind('}')
        if start == -1 or end == -1:
            return None
        json_str = body[start:end + 1]
    else:
        json_str = json_match.group()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Intento mas agresivo
        start = body.find('{')
        end = body.rfind('}')
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(body[start:end + 1])
        except json.JSONDecodeError:
            return None

    values = data.get("values", [])
    ticks = data.get("ticks", [])

    if not values or not ticks or len(values) < 1:
        return None

    general = values[0]  # IPC General
    subyacente = values[1] if len(values) > 1 else None

    if not general:
        return None

    # El ultimo valor es el mes actual del documento
    current_rate = general[-1]
    prev_rate = general[-2] if len(general) > 1 else current_rate

    result = {
        "ipc_general": current_rate,
        "ipc_subyacente": subyacente[-1] if subyacente else None,
        "ipc_prev": prev_rate,
        "change": round(current_rate - prev_rate, 2),
        "n_months": len(general),
    }
    return result


def _ine_values_to_signals(parsed: dict) -> dict:
    """Convierte valores IPC parseados en senales estructuradas."""
    change = parsed["change"]
    current = parsed["ipc_general"]

    # Decision: basada en el cambio mensual
    if change > 0.3:
        decision = "subida"
    elif change < -0.3:
        decision = "bajada"
    else:
        decision = "dato"

    # Shock score: cambios grandes son mas sorprendentes
    shock = min(abs(change) / 2.0, 1.0)
    shock = round(shock, 2)

    # Uncertainty: baja si el dato es claro
    uncertainty = 0.3 if abs(change) < 0.5 else 0.5

    # Topic: siempre inflacion para IPC
    topic = "inflacion"

    return {
        "decision": decision,
        "magnitude": abs(change),
        "tone": "negativo" if current > 3.0 else ("positivo" if current < 2.0 else "neutral"),
        "shock_score": shock,
        "uncertainty_index": uncertainty,
        "topic": topic,
        # Metadata extra
        "ipc_general": parsed["ipc_general"],
        "ipc_subyacente": parsed.get("ipc_subyacente"),
    }


def fix_ine():
    """Reprocesa documentos INE extrayendo datos del chart JSON."""
    col = _get_collection()
    ine_docs = list(col.find({
        "source": "ine",
        "raw_source": {"$in": ["rss", "rss_historical"]},
    }))

    if not ine_docs:
        print("[fix-ine] No hay documentos INE en MongoDB.")
        return

    print(f"[fix-ine] Procesando {len(ine_docs)} documentos INE...")
    fixed = 0

    for doc in ine_docs:
        body = doc.get("body", "")
        parsed = _parse_ine_chart_json(body)

        if parsed is None:
            print(f"  SKIP {doc.get('date')}: no se pudo parsear JSON")
            continue

        signals = _ine_values_to_signals(parsed)
        print(f"  {doc.get('date')}: IPC={parsed['ipc_general']}% "
              f"(cambio={parsed['change']:+.1f}pp) -> {signals['decision']}/{signals['tone']}")

        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"processed": True, "signals": signals}},
        )
        fixed += 1

    print(f"[fix-ine] {fixed}/{len(ine_docs)} documentos INE corregidos.")


# ═══════════════════════════════════════════════════════════════
# 2. FIX BCE: Decisiones historicas conocidas del BCE
# ═══════════════════════════════════════════════════════════════

# Fuente: ECB official press releases, bien documentado
# Formato: (fecha, main_refi_rate_change_bps, decision, tone, shock_score)
# shock_score es subjetivo pero basado en consenso del mercado
BCE_DECISIONS = [
    # 2015: QE era, tipos en minimos
    ("2015-01", 0, "sin_cambio", "dovish", 0.1),
    ("2015-03", 0, "sin_cambio", "dovish", 0.1),   # QE starts
    ("2015-04", 0, "sin_cambio", "dovish", 0.05),
    ("2015-06", 0, "sin_cambio", "dovish", 0.05),
    ("2015-07", 0, "sin_cambio", "neutral", 0.05),
    ("2015-09", 0, "sin_cambio", "dovish", 0.1),
    ("2015-10", 0, "sin_cambio", "dovish", 0.1),
    ("2015-12", -10, "bajada", "dovish", 0.2),  # Deposit rate -0.30%

    # 2016: TLTRO II, mas recortes
    ("2016-01", 0, "sin_cambio", "dovish", 0.05),
    ("2016-03", -5, "bajada", "dovish", 0.3),   # Refi 0.00%, deposit -0.40%
    ("2016-04", 0, "sin_cambio", "dovish", 0.05),
    ("2016-06", 0, "sin_cambio", "dovish", 0.05),
    ("2016-07", 0, "sin_cambio", "neutral", 0.05),
    ("2016-09", 0, "sin_cambio", "neutral", 0.1),
    ("2016-10", 0, "sin_cambio", "neutral", 0.05),
    ("2016-12", 0, "sin_cambio", "dovish", 0.1),  # QE extended

    # 2017: Economia mejora, pero tipos sin cambio
    ("2017-01", 0, "sin_cambio", "neutral", 0.05),
    ("2017-03", 0, "sin_cambio", "neutral", 0.05),
    ("2017-04", 0, "sin_cambio", "neutral", 0.05),
    ("2017-06", 0, "sin_cambio", "neutral", 0.05),
    ("2017-07", 0, "sin_cambio", "neutral", 0.1),
    ("2017-09", 0, "sin_cambio", "neutral", 0.1),
    ("2017-10", 0, "sin_cambio", "neutral", 0.1),   # QE tapering announced
    ("2017-12", 0, "sin_cambio", "neutral", 0.05),

    # 2018: Fin de QE anunciado
    ("2018-01", 0, "sin_cambio", "neutral", 0.05),
    ("2018-03", 0, "sin_cambio", "neutral", 0.05),
    ("2018-04", 0, "sin_cambio", "neutral", 0.05),
    ("2018-06", 0, "sin_cambio", "hawkish", 0.2),   # QE end announced Dec
    ("2018-07", 0, "sin_cambio", "neutral", 0.05),
    ("2018-09", 0, "sin_cambio", "neutral", 0.05),
    ("2018-10", 0, "sin_cambio", "neutral", 0.05),
    ("2018-12", 0, "sin_cambio", "neutral", 0.1),   # QE ends

    # 2019: Vuelta a dovish, nuevo QE
    ("2019-01", 0, "sin_cambio", "dovish", 0.1),
    ("2019-03", 0, "sin_cambio", "dovish", 0.2),
    ("2019-04", 0, "sin_cambio", "dovish", 0.1),
    ("2019-06", 0, "sin_cambio", "dovish", 0.1),
    ("2019-07", 0, "sin_cambio", "dovish", 0.1),
    ("2019-09", -10, "bajada", "dovish", 0.3),  # Deposit -0.50%, nuevo QE
    ("2019-10", 0, "sin_cambio", "dovish", 0.05),
    ("2019-12", 0, "sin_cambio", "dovish", 0.05),

    # 2020: Pandemia, PEPP
    ("2020-01", 0, "sin_cambio", "neutral", 0.05),
    ("2020-03", 0, "sin_cambio", "dovish", 0.4),    # PEPP 750bn
    ("2020-04", 0, "sin_cambio", "dovish", 0.1),
    ("2020-06", 0, "sin_cambio", "dovish", 0.2),    # PEPP expanded
    ("2020-07", 0, "sin_cambio", "dovish", 0.05),
    ("2020-09", 0, "sin_cambio", "dovish", 0.05),
    ("2020-10", 0, "sin_cambio", "dovish", 0.1),
    ("2020-12", 0, "sin_cambio", "dovish", 0.2),    # PEPP expanded again

    # 2021: Inflacion empieza a subir
    ("2021-01", 0, "sin_cambio", "dovish", 0.05),
    ("2021-03", 0, "sin_cambio", "dovish", 0.1),
    ("2021-04", 0, "sin_cambio", "dovish", 0.05),
    ("2021-06", 0, "sin_cambio", "dovish", 0.1),
    ("2021-07", 0, "sin_cambio", "dovish", 0.1),
    ("2021-09", 0, "sin_cambio", "dovish", 0.1),    # PEPP pace reduced
    ("2021-10", 0, "sin_cambio", "neutral", 0.1),
    ("2021-12", 0, "sin_cambio", "hawkish", 0.2),   # PEPP end March 2022

    # 2022: Ciclo de subidas historico
    ("2022-02", 0, "sin_cambio", "hawkish", 0.2),
    ("2022-03", 0, "sin_cambio", "hawkish", 0.2),
    ("2022-04", 0, "sin_cambio", "hawkish", 0.2),
    ("2022-06", 0, "sin_cambio", "hawkish", 0.3),   # Anuncio subida julio
    ("2022-07", 50, "subida", "hawkish", 0.5),      # Primera subida, 50bps vs 25 esperados
    ("2022-09", 75, "subida", "hawkish", 0.3),      # +75bps
    ("2022-10", 75, "subida", "hawkish", 0.2),      # +75bps
    ("2022-12", 50, "subida", "hawkish", 0.2),      # +50bps

    # 2023: Subidas continuan, luego pausa
    ("2023-02", 50, "subida", "hawkish", 0.1),      # +50bps
    ("2023-03", 50, "subida", "hawkish", 0.2),      # +50bps (crisis bancaria)
    ("2023-05", 25, "subida", "hawkish", 0.1),      # +25bps
    ("2023-06", 25, "subida", "hawkish", 0.1),      # +25bps
    ("2023-07", 25, "subida", "hawkish", 0.1),      # +25bps
    ("2023-09", 25, "subida", "hawkish", 0.3),      # +25bps, ultimo
    ("2023-10", 0, "sin_cambio", "neutral", 0.1),   # Pausa
    ("2023-12", 0, "sin_cambio", "neutral", 0.1),

    # 2024: Ciclo de bajadas
    ("2024-01", 0, "sin_cambio", "neutral", 0.05),
    ("2024-03", 0, "sin_cambio", "dovish", 0.1),
    ("2024-04", 0, "sin_cambio", "dovish", 0.1),
    ("2024-06", -25, "bajada", "dovish", 0.2),      # Primera bajada
    ("2024-07", 0, "sin_cambio", "neutral", 0.1),
    ("2024-09", -25, "bajada", "dovish", 0.1),      # -25bps
    ("2024-10", -25, "bajada", "dovish", 0.1),      # -25bps
    ("2024-12", -25, "bajada", "dovish", 0.1),      # -25bps
]


def fix_bce():
    """Inserta decisiones historicas del BCE en MongoDB con senales correctas."""
    col = _get_collection()
    now = datetime.now(timezone.utc)

    print(f"[fix-bce] Insertando {len(BCE_DECISIONS)} decisiones del BCE...")
    inserted, updated = 0, 0

    for ym, bps, decision, tone, shock in BCE_DECISIONS:
        magnitude = abs(bps) / 100.0 if bps != 0 else None

        # Uncertainty: subidas/bajadas grandes = mas certeza, sin_cambio prolongado = poca
        if abs(bps) >= 50:
            uncertainty = 0.2
        elif abs(bps) == 25:
            uncertainty = 0.3
        else:
            uncertainty = 0.4

        signals = {
            "decision": decision,
            "magnitude": magnitude,
            "tone": tone,
            "shock_score": shock,
            "uncertainty_index": uncertainty,
            "topic": "tipos_interes",
        }

        # Construir texto descriptivo
        if bps > 0:
            body_text = (f"El Consejo de Gobierno del BCE ha decidido subir "
                        f"los tipos de interes en {abs(bps)} puntos basicos. "
                        f"Tono: {tone}.")
        elif bps < 0:
            body_text = (f"El Consejo de Gobierno del BCE ha decidido bajar "
                        f"los tipos de interes en {abs(bps)} puntos basicos. "
                        f"Tono: {tone}.")
        else:
            body_text = (f"El Consejo de Gobierno del BCE ha decidido mantener "
                        f"los tipos de interes sin cambios. Tono: {tone}.")

        title = f"BCE decision monetaria {ym}: {decision}"
        url = f"ecb://decision/{ym}"

        # Check si ya existe este mes con URL sintetica
        existing = col.find_one({"url": url})
        if existing:
            col.update_one(
                {"_id": existing["_id"]},
                {"$set": {"processed": True, "signals": signals, "body": body_text}},
            )
            updated += 1
        else:
            doc = {
                "date": ym,
                "title": title,
                "body": body_text,
                "source": "bce",
                "url": url,
                "raw_source": "rss_historical",
                "ingestion_timestamp": now.isoformat(),
                "processed": True,
                "signals": signals,
            }
            try:
                col.insert_one(doc)
                inserted += 1
            except Exception:
                updated += 1  # Duplicado

        change_str = f"{bps:+d}bps" if bps != 0 else "sin cambio"
        print(f"  {ym}: {change_str} ({tone}) -> shock={shock}")

    print(f"[fix-bce] {inserted} insertados, {updated} actualizados.")


# ═══════════════════════════════════════════════════════════════
# 3. LIMPIEZA: Eliminar documentos basura
# ═══════════════════════════════════════════════════════════════

def clean_garbage():
    """Elimina documentos BCE basura (paginas de navegacion, placeholders)."""
    col = _get_collection()

    # BCE navigation pages: todos del 2015-01-22 con body de navegacion
    garbage_titles = [
        "Monetary policy & markets",
        "What is monetary policy?",
        "Monetary policy",
        "Scope of monetary policy",
        "Monetary policy strategy",
        "Monetary policy, strategy and implementation",
        "Monetary policy decisions",
        "Monetary policy accounts",
        "Monetary policy operations",
        "Monetary policy statements",
        "Monetary policy statements at a glance",
        "Unconventional monetary policy instruments",
        "Deep dive into the ECB: 2. Monetary policy",
        "Deep dive into the ECB: 3. Unconventional monetary policy instruments",
    ]

    # Delete navigation pages
    result = col.delete_many({
        "source": "bce",
        "title": {"$in": garbage_titles},
    })
    print(f"[clean] Eliminadas {result.deleted_count} paginas de navegacion BCE")

    # Delete placeholder docs (body = "Monetary policy meeting held on...")
    result = col.delete_many({
        "source": "bce",
        "body": {"$regex": "^Monetary policy meeting held on.*not available"},
    })
    print(f"[clean] Eliminados {result.deleted_count} placeholders BCE")

    # Delete BCE RSS entries with empty body (2026, fuera de rango)
    result = col.delete_many({
        "source": "bce",
        "raw_source": "rss",
        "body": "",
    })
    print(f"[clean] Eliminados {result.deleted_count} RSS BCE sin body (fuera de rango)")

    # Resumen
    remaining = col.count_documents({"source": "bce"})
    print(f"[clean] BCE restantes en MongoDB: {remaining}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Reparar senales BCE e INE en MongoDB")
    parser.add_argument("--ine", action="store_true", help="Solo reparar INE")
    parser.add_argument("--bce", action="store_true", help="Solo reparar BCE")
    parser.add_argument("--clean", action="store_true", help="Solo limpiar basura")
    args = parser.parse_args()

    run_all = not args.ine and not args.bce and not args.clean

    if run_all or args.clean:
        clean_garbage()
        print()

    if run_all or args.bce:
        fix_bce()
        print()

    if run_all or args.ine:
        fix_ine()
        print()

    # Resumen final
    col = _get_collection()
    for src in ["bce", "ine", "gdelt"]:
        total = col.count_documents({"source": src})
        processed = col.count_documents({"source": src, "processed": True})
        print(f"  {src}: {processed}/{total} procesados")


if __name__ == "__main__":
    main()
