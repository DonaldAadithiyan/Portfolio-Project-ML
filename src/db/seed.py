import logging
import math

import pandas as pd

from src.db.db import get_client

logger = logging.getLogger(__name__)


_INT_COLS = {
    "depot_id",
    "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
    "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
    "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
}


def _clean(val, col: str = ""):
    """Convert NaN/inf to None; cast integer columns to int for JSON serialisation."""
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    if col in _INT_COLS:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def seed_depots(depots_cfg: list[dict]) -> dict[str, int]:
    """Upsert depot rows (idempotent). Returns {name: depot_id} map."""
    sb = get_client()
    rows = [
        {
            "name": d["name"],
            "district": d.get("district"),
            "province": d.get("province"),
            "latitude": d["lat"],
            "longitude": d["lon"],
            "pop_weight": d["pop_weight"],
        }
        for d in depots_cfg
    ]
    sb.table("tc_depots").upsert(rows, on_conflict="name").execute()

    result = sb.table("tc_depots").select("name,depot_id").execute()
    depot_map = {r["name"]: r["depot_id"] for r in result.data}
    logger.info("[SEED] Depots seeded: %d", len(depot_map))
    return depot_map


def seed_demand_panel(panel_path: str, depot_map: dict[str, int]) -> int:
    """Bulk-upsert panel rows into tc_demand_panel. Idempotent (ON CONFLICT DO NOTHING)."""
    df = pd.read_csv(panel_path, parse_dates=["week_start"])
    df["depot_id"] = df["depot"].map(depot_map)

    missing = df[df["depot_id"].isna()]["depot"].unique()
    if len(missing):
        raise ValueError(f"Unknown depots in panel: {missing}")

    df = df.drop_duplicates(subset=["depot_id", "week_start"], keep="last")

    col_order = [
        "depot_id", "week_start", "demand_tonnes", "sales_tonnes", "production_tonnes",
        "precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean",
        "gdp_lka", "lending_rate", "cbsl_pmi_construction", "govt_consumption",
        "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
        "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
        "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = None

    sb = get_client()
    records = []
    for _, row in df[col_order].iterrows():
        rec = {}
        for col in col_order:
            val = row[col]
            if col == "week_start":
                rec[col] = val.date().isoformat() if hasattr(val, "date") else str(val)
            elif not pd.notna(val):
                rec[col] = None
            else:
                rec[col] = _clean(val, col)
        records.append(rec)

    batch_size = 500
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        sb.table("tc_demand_panel").upsert(
            batch, on_conflict="depot_id,week_start"
        ).execute()
        inserted += len(batch)
        logger.info("[SEED] demand_panel progress: %d / %d", inserted, len(records))

    logger.info("[SEED] demand_panel rows upserted: %d", len(records))
    return len(records)
