"""
news_to_features_global.py
---------------------------
Orquestador del pipeline MCP semántico global. Pasos:

  --scrape [fomc|ecb|bls|all]
      Descarga comunicados de las fuentes seleccionadas.

  --process [fomc|ecb|bls|all]
      Extrae señales LLM (Qwen3:4b) de los documentos no procesados.

  --correlate
      Calcula correlaciones de las 9 señales MCP con:
        1. cpi_global_rate(t+1)
        2. Residuos de Chronos-2 C1_inst_global (horizonte h=1)
      Criterio de parada temprana: si todas |corr residuos| < 0.15,
      no lanzar modelos completos.

  --build-c1
      Agrega señales a frecuencia mensual, aplica shift +1 (leakage),
      merge con features_c1_global_institutional.parquet,
      exporta features_c1_global_full.parquet.

Control de leakage:
  Shift +1 mensual en TODAS las señales. La señal del mes t
  (p.ej. FOMC del 15 marzo) entra como exógena a partir de abril
  en el rolling backtesting. Esto se replica en build-c1.

Esquema final features_c1_global_full.parquet:
  [todas las columnas de features_c1_global_institutional.parquet]
  + fomc_hawkish_score, fomc_surprise_score, fomc_forward_guidance_num
  + ecb_hawkish_score, ecb_surprise_score, ecb_forward_guidance_num
  + us_cpi_surprise_score, us_cpi_direction_num, us_cpi_components_pressure
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
MONOREPO = ROOT.parent
sys.path.insert(0, str(MONOREPO))
sys.path.insert(0, str(Path(__file__).parent))

from shared.constants import DATE_TEST_END

RAW_BASE  = ROOT / "data" / "raw" / "global_raw"
DATA_PROC = ROOT / "data" / "processed"

INST_FEATURES = DATA_PROC / "features_c1_global_institutional.parquet"
MCP_SIGNALS   = DATA_PROC / "mcp_signals_global.parquet"
FULL_FEATURES = DATA_PROC / "features_c1_global_full.parquet"

IDX = pd.date_range("2002-01-01", "2024-12-01", freq="MS")

# Columnas MCP numéricas finales (9 señales semánticas)
FOMC_COLS = ["fomc_hawkish_score", "fomc_surprise_score", "fomc_forward_guidance_num"]
ECB_COLS  = ["ecb_hawkish_score",  "ecb_surprise_score",  "ecb_forward_guidance_num"]
BLS_COLS  = ["us_cpi_surprise_score", "us_cpi_direction_num", "us_cpi_components_pressure"]
MCP_COLS  = FOMC_COLS + ECB_COLS + BLS_COLS

GUIDANCE_MAP = {"subida": 1.0, "neutral": 0.0, "bajada": -1.0}
CPI_DIR_MAP  = {"aceleracion": 1.0, "estable": 0.0, "desaceleracion": -1.0}

# Defaults para meses sin documento
MCP_DEFAULTS = {
    "fomc_hawkish_score": 0.5, "fomc_surprise_score": 0.0, "fomc_forward_guidance_num": 0.0,
    "ecb_hawkish_score": 0.5,  "ecb_surprise_score": 0.0,  "ecb_forward_guidance_num": 0.0,
    "us_cpi_surprise_score": 0.0, "us_cpi_direction_num": 0.0, "us_cpi_components_pressure": 0.5,
}


# ==================================================================
# PASO 1: Scraping
# ==================================================================

def scrape(sources: list[str]):
    if "fomc" in sources or "all" in sources:
        from fetch_fomc_historical import fetch_fomc
        fetch_fomc()

    if "ecb" in sources or "all" in sources:
        from fetch_ecb_press_historical import fetch_ecb_press
        fetch_ecb_press()

    if "bls" in sources or "all" in sources:
        from fetch_bls_cpi_historical import fetch_bls_cpi
        fetch_bls_cpi()


# ==================================================================
# PASO 2: Procesamiento LLM
# ==================================================================

def _load_pending_docs(source_dir: Path, source_key: str) -> list[dict]:
    """Carga todos los documentos JSON no procesados de una fuente."""
    pending = []
    if not source_dir.exists():
        return pending
    for f in sorted(source_dir.glob("*.json")):
        try:
            docs = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(docs, list):
                continue
            for doc in docs:
                if not doc.get("processed", False):
                    doc["_file"] = str(f)
                    pending.append(doc)
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")
    return pending


def _update_doc_file(doc: dict, signals: dict):
    """Actualiza el JSON en disco con las señales extraídas."""
    file_path = Path(doc.get("_file", ""))
    if not file_path.exists():
        return
    try:
        docs = json.loads(file_path.read_text(encoding="utf-8"))
        for d in docs:
            if d.get("url") == doc.get("url") and d.get("date") == doc.get("date"):
                d["processed"] = True
                d["signals"] = signals
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  [WARN] No se pudo actualizar {file_path.name}: {e}")


def process(sources: list[str]):
    from agent_extractor_global import extract_signals

    source_map = {
        "fomc":      (RAW_BASE / "fomc",      "fomc"),
        "ecb":       (RAW_BASE / "ecb_press",  "ecb_press"),
        "bls":       (RAW_BASE / "bls_cpi",   "bls_cpi"),
    }

    to_process = (
        list(source_map.items())
        if "all" in sources
        else [(k, v) for k, v in source_map.items() if k in sources]
    )

    for src_name, (src_dir, src_key) in to_process:
        print(f"\n{'='*50}")
        print(f"LLM Extraction — {src_name.upper()}")
        print(f"{'='*50}")

        pending = _load_pending_docs(src_dir, src_key)
        if not pending:
            print(f"  Sin documentos pendientes en {src_dir}")
            continue

        print(f"  {len(pending)} documentos por procesar con qwen3:4b")

        for i, doc in enumerate(pending):
            text = f"{doc.get('title', '')} {doc.get('body', '')}"
            ym   = str(doc.get("date", ""))[:7]

            print(f"  [{i+1:3d}/{len(pending)}] {ym} {src_name} ...", end=" ", flush=True)
            signals = extract_signals(text, source=src_key)
            _update_doc_file(doc, signals)
            key_score = signals.get("fomc_hawkish_score",
                        signals.get("ecb_hawkish_score",
                        signals.get("us_cpi_surprise_score", 0.0)))
            print(f"done ({key_score:.2f})")

        print(f"  Completado: {len(pending)} docs procesados")


# ==================================================================
# PASO 3: Construcción del parquet de señales
# ==================================================================

def _aggregate_source(src_dir: Path, source_key: str) -> pd.DataFrame:
    """
    Lee JSONs procesados de una fuente y agrega a frecuencia mensual.
    Devuelve DataFrame con index=date y columnas de señales.
    """
    rows = []
    if not src_dir.exists():
        return pd.DataFrame()

    for f in sorted(src_dir.glob("*.json")):
        try:
            docs = json.loads(f.read_text(encoding="utf-8"))
            for doc in docs:
                if not doc.get("processed"):
                    continue
                signals = doc.get("signals", {})
                date_str = str(doc.get("date", ""))[:7]  # YYYY-MM
                if not date_str or len(date_str) < 7:
                    continue

                row = {"date": date_str}

                if source_key == "fomc":
                    row["fomc_hawkish_score"]  = float(signals.get("fomc_hawkish_score", 0.5))
                    row["fomc_surprise_score"] = float(signals.get("fomc_surprise_score", 0.0))
                    row["fomc_forward_guidance_num"] = GUIDANCE_MAP.get(
                        signals.get("fomc_forward_guidance", "neutral"), 0.0
                    )
                elif source_key == "ecb_press":
                    row["ecb_hawkish_score"]  = float(signals.get("ecb_hawkish_score", 0.5))
                    row["ecb_surprise_score"] = float(signals.get("ecb_surprise_score", 0.0))
                    row["ecb_forward_guidance_num"] = GUIDANCE_MAP.get(
                        signals.get("ecb_forward_guidance", "neutral"), 0.0
                    )
                elif source_key == "bls_cpi":
                    row["us_cpi_surprise_score"]      = float(signals.get("us_cpi_surprise_score", 0.0))
                    row["us_cpi_direction_num"]         = CPI_DIR_MAP.get(
                        signals.get("us_cpi_direction", "estable"), 0.0
                    )
                    row["us_cpi_components_pressure"] = float(signals.get("us_cpi_components_pressure", 0.5))

                rows.append(row)
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Si hay varios docs por mes, tomar la media
    num_cols = [c for c in df.columns if c != "date"]
    df = df.groupby("date")[num_cols].mean().reset_index()
    df["date"] = pd.to_datetime(df["date"] + "-01")
    df = df.set_index("date")
    return df


def build_signals_parquet() -> pd.DataFrame:
    """Agrega las 3 fuentes a mensual y exporta mcp_signals_global.parquet."""
    print("\n[build] Agregando señales MCP a frecuencia mensual...")

    df_fomc = _aggregate_source(RAW_BASE / "fomc",     "fomc")
    df_ecb  = _aggregate_source(RAW_BASE / "ecb_press", "ecb_press")
    df_bls  = _aggregate_source(RAW_BASE / "bls_cpi",  "bls_cpi")

    print(f"  FOMC:    {len(df_fomc)} meses con señales")
    print(f"  ECB:     {len(df_ecb)}  meses con señales")
    print(f"  BLS CPI: {len(df_bls)} meses con señales")

    # Base: índice completo 2002-2024
    df = pd.DataFrame(index=IDX)
    df.index.name = "date"

    for col in MCP_COLS:
        df[col] = MCP_DEFAULTS[col]

    # Merge fuentes
    for df_src in [df_fomc, df_ecb, df_bls]:
        if df_src.empty:
            continue
        for col in df_src.columns:
            if col in df.columns:
                df.loc[df_src.index, col] = df_src[col]

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MCP_SIGNALS)
    print(f"  Guardado: {MCP_SIGNALS}")
    return df


# ==================================================================
# PASO 3b: Correlaciones + criterio de parada temprana
# ==================================================================

def correlate(
    stop_threshold: float = 0.15,
    full_threshold: float = 0.20,
) -> bool:
    """
    Calcula correlaciones por fuente usando SOLO los meses con datos reales
    (no los defaults — evita dilución).

    Por fuente:
      • corr(CPI t+1)       — poder predictivo directo
      • corr(residuos h=1)  — poder predictivo incremental sobre Chronos-2

    Criterio de decisión:
      • Si max |corr residuos| de TODAS las fuentes < stop_threshold (0.15)
        -> PARAR, hallazgo negativo confirmado.
      • Si alguna fuente tiene max |corr residuos| > full_threshold (0.20)
        -> CONTINUAR con esa fuente (procesar histórico completo).

    Returns True si alguna fuente supera full_threshold.
    """
    print(f"\n{'='*65}")
    print("CORRELACIONES MCP — Señales vs CPI y Residuos Chronos-2")
    print(f"Umbrales: stop={stop_threshold}  |  continuar={full_threshold}")
    print(f"{'='*65}")

    # Cargar CPI
    cpi_path = DATA_PROC / "cpi_global_monthly.parquet"
    if not cpi_path.exists():
        print(f"[!] No se encuentra {cpi_path}")
        return True

    cpi = pd.read_parquet(cpi_path)
    if "date" in cpi.columns:
        cpi = cpi.set_index("date")
    cpi.index = pd.to_datetime(cpi.index)
    target_lead = cpi["cpi_global_rate"].shift(-1)

    # Cargar residuos Chronos-2 h=1
    preds_path = ROOT / "08_results" / "chronos2_C1_inst_global_predictions.parquet"
    residuals = None
    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        preds_h1 = preds[preds["horizon"] == 1].copy()
        preds_h1["fc_date"] = pd.to_datetime(preds_h1["fc_date"])
        residuals = preds_h1.groupby("fc_date")["error"].mean()
        print(f"  Residuos Chronos-2 C1_inst h=1: {len(residuals)} obs ({residuals.index.min().date()} - {residuals.index.max().date()})")
    else:
        print("  [WARN] Chronos-2 predictions no encontradas")

    # Definición de fuentes: (nombre, dir, source_key, señales)
    source_defs = [
        ("FOMC",    RAW_BASE / "fomc",      "fomc",     FOMC_COLS),
        ("ECB",     RAW_BASE / "ecb_press",  "ecb_press", ECB_COLS),
        ("BLS CPI", RAW_BASE / "bls_cpi",   "bls_cpi",  BLS_COLS),
    ]

    sep = "-" * 72
    all_results = {}
    sources_above_full = []

    for src_name, src_dir, src_key, src_cols in source_defs:
        # Obtener señales SOLO de meses con datos reales (no defaults)
        df_src = _aggregate_source(src_dir, src_key)

        if df_src.empty:
            print(f"\n  [{src_name}] Sin datos procesados aún — omitiendo")
            continue

        n_months = len(df_src)
        date_range = f"{df_src.index.min().date()} -> {df_src.index.max().date()}"
        print(f"\n  -- {src_name} ({n_months} meses con datos reales: {date_range}) --")
        print(f"  {sep}")
        print(f"  {'Señal':<35} {'corr(CPI t+1)':>14} {'corr(residuos)':>14}  N")
        print(f"  {sep}")

        src_max_res = 0.0
        src_results = {}

        for col in src_cols:
            if col not in df_src.columns:
                continue

            signal = df_src[col].dropna()

            # corr con CPI t+1 (solo meses con señal real)
            common_cpi = signal.index.intersection(target_lead.dropna().index)
            corr_cpi = (
                signal.loc[common_cpi].corr(target_lead.loc[common_cpi])
                if len(common_cpi) >= 8 else np.nan
            )

            # corr con residuos
            corr_res = np.nan
            n_res = 0
            if residuals is not None:
                common_res = signal.index.intersection(residuals.dropna().index)
                n_res = len(common_res)
                if n_res >= 8:
                    corr_res = signal.loc[common_res].corr(residuals.loc[common_res])

            src_results[col] = {"corr_cpi": corr_cpi, "corr_residuals": corr_res, "n": n_res}
            all_results[col] = src_results[col]

            if not np.isnan(corr_res):
                src_max_res = max(src_max_res, abs(corr_res))

            mark = " <-- !" if (not np.isnan(corr_res) and abs(corr_res) >= stop_threshold) else ""
            cpi_str = f"{corr_cpi:>14.3f}" if not np.isnan(corr_cpi) else f"{'n/a':>14}"
            res_str = f"{corr_res:>14.3f}" if not np.isnan(corr_res) else f"{'n/a':>14}"
            print(f"  {col:<35} {cpi_str} {res_str}  {n_res:3d}{mark}")

        print(f"  {sep}")
        verdict = ""
        if src_max_res >= full_threshold:
            verdict = f"CONTINUAR con {src_name} (max |corr res|={src_max_res:.3f} >= {full_threshold})"
            sources_above_full.append(src_name)
        elif src_max_res >= stop_threshold:
            verdict = f"BORDERLINE {src_name} (max |corr res|={src_max_res:.3f})"
        else:
            verdict = f"PARAR {src_name} (max |corr res|={src_max_res:.3f} < {stop_threshold})"
        print(f"  -> {verdict}")

    # Guardar
    corr_path = DATA_PROC / "mcp_correlations_global.json"
    with open(corr_path, "w") as f:
        json.dump({
            k: {k2: (float(v2) if isinstance(v2, float) and not np.isnan(v2) else None)
                for k2, v2 in v.items()}
            for k, v in all_results.items()
        }, f, indent=2)

    # Veredicto global
    print(f"\n{'='*65}")
    if sources_above_full:
        print(f"DECISIÓN: CONTINUAR con {', '.join(sources_above_full)}")
        print(f"  Procesar histórico completo de esas fuentes y lanzar modelos C1_mcp.")
        print(f"{'='*65}")
        return True
    else:
        max_any = max(
            (abs(v["corr_residuals"]) for v in all_results.values()
             if v["corr_residuals"] is not None and not np.isnan(v["corr_residuals"])),
            default=0.0,
        )
        print(f"DECISIÓN: PARAR — hallazgo negativo confirmado (max |corr res|={max_any:.3f})")
        print(f"  Ninguna fuente MCP supera el umbral de continuación ({full_threshold}).")
        print(f"  Documentar en tesis: la capa semántica no aporta poder predictivo")
        print(f"  incremental sobre Chronos-2 C1_institutional.")
        print(f"{'='*65}")
        return False


# ==================================================================
# PASO 4: Build features_c1_global_full.parquet
# ==================================================================

def build_c1():
    """
    Merge features_c1_global_institutional + señales MCP (con shift +1).
    Exporta features_c1_global_full.parquet.
    """
    print(f"\n{'='*60}")
    print("BUILD — features_c1_global_full.parquet")
    print(f"{'='*60}")

    # Construir señales si no existen
    if not MCP_SIGNALS.exists():
        print("[build] Construyendo mcp_signals_global.parquet...")
        build_signals_parquet()

    df_mcp = pd.read_parquet(MCP_SIGNALS)
    df_mcp.index = pd.to_datetime(df_mcp.index)

    # Shift +1: señales del mes t entran en t+1 (anti-leakage)
    df_mcp_shifted = df_mcp.copy()
    df_mcp_shifted[MCP_COLS] = df_mcp_shifted[MCP_COLS].shift(1)

    # Rellenar NaN de la primera fila tras el shift
    for col in MCP_COLS:
        df_mcp_shifted[col] = df_mcp_shifted[col].fillna(MCP_DEFAULTS[col])

    print(f"  [leakage fix] señales MCP shifteadas +1 mes")

    if not INST_FEATURES.exists():
        print(f"[!] No existe {INST_FEATURES}")
        print("    Exportando solo señales MCP como features_c1_global_full.parquet")
        df_mcp_shifted.to_parquet(FULL_FEATURES)
        print(f"  Guardado: {FULL_FEATURES}")
        return

    df_inst = pd.read_parquet(INST_FEATURES)
    df_inst.index = pd.to_datetime(df_inst.index)

    # Merge
    df_full = df_inst.copy()
    for col in MCP_COLS:
        df_full[col] = df_mcp_shifted[col].reindex(df_full.index)
        df_full[col] = df_full[col].fillna(MCP_DEFAULTS[col])

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(FULL_FEATURES)
    print(f"\n  features_c1_global_full.parquet:")
    print(f"    Shape: {df_full.shape}")
    print(f"    Columnas MCP: {MCP_COLS}")
    print(f"    NaN totales: {df_full.isna().sum().sum()}")
    print(f"    Guardado: {FULL_FEATURES}")


# ==================================================================
# CLI
# ==================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline MCP semántico global: scraping, LLM, correlaciones, features"
    )
    parser.add_argument(
        "--scrape", nargs="+", choices=["fomc", "ecb", "bls", "all"],
        metavar="SOURCE",
        help="Descargar fuentes: fomc ecb bls all",
    )
    parser.add_argument(
        "--process", nargs="+", choices=["fomc", "ecb", "bls", "all"],
        metavar="SOURCE",
        help="Procesar con LLM (Qwen3:4b)",
    )
    parser.add_argument(
        "--correlate", action="store_true",
        help="Calcular correlaciones + criterio de parada temprana",
    )
    parser.add_argument(
        "--build-c1", action="store_true",
        help="Exportar features_c1_global_full.parquet",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Umbral para criterio de parada temprana (default: 0.15)",
    )

    args = parser.parse_args()
    did_something = False

    if args.scrape:
        scrape(args.scrape)
        did_something = True

    if args.process:
        process(args.process)
        build_signals_parquet()  # rebuild parquet después de procesar
        did_something = True

    if args.correlate:
        correlate(stop_threshold=args.threshold, full_threshold=args.threshold + 0.05)
        did_something = True

    if args.build_c1:
        build_c1()
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()
