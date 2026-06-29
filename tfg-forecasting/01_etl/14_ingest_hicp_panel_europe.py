"""
14_ingest_hicp_panel_europe.py - per-country euro-area HICP panel (Phase B)
---------------------------------------------------------------------------
Same ECB SDMX source and series structure as 11_ingest_hicp_europe.py, but
fetched for each euro-area member instead of the U2 aggregate:

    ICP.M.<REF_AREA>.N.000000.4.INX   (monthly, all-items, price-level index)

Used to expand the forecasting cross-section from 3 series (Spain/Global/Europe)
to ~20 countries so the pooled (panel) predictive-accuracy test gains real
statistical power (Phase B). The exogenous covariates stay the *shared euro-area*
set already built in features_c1_europe.parquet (EPU Europe, Brent, ECB rate,
ESI, EUR/USD): these are area-wide drivers common to every euro member, so no
per-country covariate sourcing is needed.

Only countries with COMPLETE monthly coverage 2002-01..2024-12 are kept (so every
panel unit shares the same origins and MASE base window).

Output:
  data/processed/hicp_panel_europe.parquet   (long: country, date, hicp_index)
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

OUT_PQ = ROOT / "data" / "processed" / "hicp_panel_europe.parquet"
START, END = "2002-01", "2024-12"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TFG-Academic-Research/1.0)",
    "Accept": "text/csv",
}
URL = ("https://data-api.ecb.europa.eu/service/data/ICP/"
       "M.{area}.N.000000.4.INX?format=csvdata&startPeriod={s}&endPeriod={e}")

# Euro-area members (ECB REF_AREA codes). Later joiners are included; those
# without full 2002 coverage are dropped by the completeness filter below.
AREAS = ["AT", "BE", "DE", "ES", "FI", "FR", "GR", "IE", "IT", "LU", "NL", "PT",
         "SI", "CY", "MT", "SK", "EE", "LV", "LT"]

EXPECTED_MONTHS = pd.date_range(START + "-01", END + "-01", freq="MS")


def fetch_country(area: str) -> pd.DataFrame | None:
    url = URL.format(area=area, s=START, e=END)
    try:
        r = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        logger.warning("  %s: fetch failed (%s)", area, type(e).__name__)
        return None
    d = pd.read_csv(StringIO(r.text))
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(d.columns):
        logger.warning("  %s: unexpected columns", area)
        return None
    d = d[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "date", "OBS_VALUE": "hicp_index"})
    d["date"] = pd.to_datetime(d["date"].astype(str).str.strip() + "-01")
    d["hicp_index"] = pd.to_numeric(d["hicp_index"], errors="coerce")
    d = d.dropna(subset=["hicp_index"]).sort_values("date")
    d = d[(d["date"] >= EXPECTED_MONTHS[0]) & (d["date"] <= EXPECTED_MONTHS[-1])]
    d["country"] = area
    return d


def main() -> None:
    logger.info("=" * 60)
    logger.info("ETL - euro-area HICP panel (ECB SDMX, %d candidate areas)", len(AREAS))
    logger.info("=" * 60)

    kept, dropped = [], []
    for area in AREAS:
        d = fetch_country(area)
        if d is None:
            dropped.append((area, "fetch/parse"))
        elif len(d) != len(EXPECTED_MONTHS):
            dropped.append((area, f"incomplete ({len(d)}/{len(EXPECTED_MONTHS)})"))
            logger.info("  %s: %d/%d months -> dropped", area, len(d), len(EXPECTED_MONTHS))
        else:
            kept.append(d)
            logger.info("  %s: OK (%d months, %.1f..%.1f)",
                        area, len(d), d["hicp_index"].min(), d["hicp_index"].max())
        time.sleep(0.4)  # be polite to the API

    if not kept:
        logger.error("No countries with complete coverage; aborting.")
        raise SystemExit(1)

    panel = pd.concat(kept, ignore_index=True)[["country", "date", "hicp_index"]]
    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PQ, index=False)

    countries = sorted(panel["country"].unique())
    logger.info("\n" + "=" * 60)
    logger.info("Saved: %s", OUT_PQ)
    logger.info("Countries kept (%d): %s", len(countries), ", ".join(countries))
    if dropped:
        logger.info("Dropped (%d): %s", len(dropped),
                    ", ".join(f"{a}({why})" for a, why in dropped))
    logger.info("Rows: %d  | months/country: %d", len(panel), len(EXPECTED_MONTHS))


if __name__ == "__main__":
    main()
