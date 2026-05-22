import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_weather_for_depot(depot_name: str, weather_dir: str) -> pd.DataFrame:
    fname = depot_name.lower().replace(" ", "_") + ".csv"
    path = os.path.join(weather_dir, fname)
    if not os.path.exists(path):
        logger.warning("[FEATURES] Weather file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["week_start"])
    return df


def _add_lag_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add demand lag and rolling features per depot, ordered by week."""
    lag_weeks = cfg["features"]["lag_weeks"]
    rolling_windows = cfg["features"]["rolling_windows"]

    df = df.sort_values(["depot", "week_start"]).copy()

    for lag in lag_weeks:
        df[f"demand_lag_{lag}"] = df.groupby("depot")["demand_tonnes"].shift(lag)

    for window in rolling_windows:
        df[f"demand_rolling_mean_{window}"] = (
            df.groupby("depot")["demand_tonnes"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
    df["demand_rolling_std_4"] = (
        df.groupby("depot")["demand_tonnes"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
    )
    return df


def build_features(cfg: dict) -> pd.DataFrame:
    """
    Join depot split + weather + economics + calendar into final panel.
    Add lag features and interaction terms.
    Writes: data/processed/panel_modelling.csv
    Returns the DataFrame.
    """
    out_path = os.path.join(cfg["paths"]["processed"], "panel_modelling.csv")
    os.makedirs(cfg["paths"]["processed"], exist_ok=True)

    # ── 1. Load depot split ───────────────────────────────────
    depot_path = os.path.join(cfg["paths"]["interim"], "kaggle_depot_split.csv")
    panel = pd.read_csv(depot_path, parse_dates=["week_start"])
    logger.info("[FEATURES] Depot split loaded: %d rows", len(panel))

    # ── 2. Join weather per depot ─────────────────────────────
    weather_dir = cfg["paths"]["raw_weather"]
    weather_frames = []
    for depot_name in panel["depot"].unique():
        w = _load_weather_for_depot(depot_name, weather_dir)
        if w.empty:
            continue
        w["depot"] = depot_name
        weather_frames.append(w)

    if weather_frames:
        weather_all = pd.concat(weather_frames, ignore_index=True)
        panel = panel.merge(weather_all, on=["week_start", "depot"], how="left")
        logger.info("[FEATURES] Weather joined")

    # ── 3. Join economics (same for all depots) ───────────────
    econ_path = os.path.join(cfg["paths"]["raw_economic"], "worldbank_lka.csv")
    if os.path.exists(econ_path):
        econ = pd.read_csv(econ_path, parse_dates=["week_start"])
        econ_cols = [c for c in ["gdp_lka", "lending_rate", "govt_consumption"] if c in econ.columns]
        panel = panel.merge(econ[["week_start"] + econ_cols], on="week_start", how="left")
        logger.info("[FEATURES] Economics joined")

    # ── 4. Join CBSL PMI ──────────────────────────────────────
    pmi_path = os.path.join(cfg["paths"]["raw_economic"], "worldbank_lka.csv")
    # PMI is already optionally in worldbank_lka if merged; check for a separate file
    pmi_sep_path = os.path.join(cfg["paths"]["raw_economic"], "cbsl_pmi_weekly.csv")
    if os.path.exists(pmi_sep_path):
        pmi = pd.read_csv(pmi_sep_path, parse_dates=["week_start"])
        panel = panel.merge(pmi[["week_start", "cbsl_pmi_construction"]], on="week_start", how="left")

    # ── 5. Join calendar ─────────────────────────────────────
    cal_path = os.path.join(cfg["paths"]["raw_calendar"], "lka_calendar.csv")
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path, parse_dates=["week_start"])
        cal_cols = [c for c in cal.columns if c != "week_start"]
        panel = panel.merge(cal[["week_start"] + cal_cols], on="week_start", how="left")
        logger.info("[FEATURES] Calendar joined")

    # ── 6. Lag features ───────────────────────────────────────
    panel = _add_lag_features(panel, cfg)
    logger.info("[FEATURES] Lag features added")

    # ── 7. Interaction features ───────────────────────────────
    if "precip_sum" in panel.columns and "is_sw_monsoon" in panel.columns:
        panel["precip_x_monsoon"] = panel["precip_sum"] * panel["is_sw_monsoon"]

    if "post_holiday_lag_1" in panel.columns and "demand_rolling_mean_4" in panel.columns:
        panel["post_holiday_demand_boost"] = (
            panel["post_holiday_lag_1"] * panel["demand_rolling_mean_4"]
        )

    # ── 8. Encode depot as integer ────────────────────────────
    depot_names = sorted(panel["depot"].unique())
    depot_enc = {n: i for i, n in enumerate(depot_names)}
    panel["depot_enc"] = panel["depot"].map(depot_enc)

    # ── 9. Drop columns that cause leakage ───────────────────
    drop_cols = cfg["features"].get("drop_cols", [])
    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns], errors="ignore")

    # ── 10. Sort and save ─────────────────────────────────────
    panel = panel.sort_values(["depot", "week_start"]).reset_index(drop=True)
    panel.to_csv(out_path, index=False)
    logger.info("[FEATURES] Final panel: %d rows × %d cols -> %s",
                len(panel), len(panel.columns), out_path)
    return panel


def rebuild_lag_features_for_depots(depots: list[str], cfg: dict) -> pd.DataFrame:
    """
    Reload the panel from DB, rebuild lag features for specified depots.
    Used by the retraining path after new sales_actuals rows are inserted.
    """
    from src.db.db import get_client

    sb = get_client()

    # Fetch all demand panel rows with pagination (Supabase default limit is 1000)
    select_cols = (
        "week_start,depot_id,demand_tonnes,precip_sum,rain_sum,temp_mean,"
        "humidity_mean,cloud_cover_mean,gdp_lka,lending_rate,"
        "cbsl_pmi_construction,govt_consumption,is_sw_monsoon,is_ne_monsoon,"
        "is_dry_season,is_sinhala_tamil_new_year,is_vesak,is_christmas_week,"
        "post_holiday_lag_1,post_holiday_lag_2,is_year_end_quarter,data_source"
    )
    all_rows = []
    page_size = 1000
    start = 0
    while True:
        result = sb.table("tc_demand_panel").select(select_cols).range(
            start, start + page_size - 1
        ).execute()
        all_rows.extend(result.data)
        if len(result.data) < page_size:
            break
        start += page_size
    df = pd.DataFrame(all_rows)

    # Fetch depot id→name map and join
    depots_result = sb.table("tc_depots").select("depot_id,name").execute()
    depot_map = {r["depot_id"]: r["name"] for r in depots_result.data}
    df["depot"] = df["depot_id"].map(depot_map)
    df = df.drop(columns=["depot_id"])

    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["depot", "week_start"]).reset_index(drop=True)
    df = _add_lag_features(df, cfg)

    if "precip_sum" in df.columns and "is_sw_monsoon" in df.columns:
        df["precip_x_monsoon"] = df["precip_sum"] * df["is_sw_monsoon"]
    if "post_holiday_lag_1" in df.columns and "demand_rolling_mean_4" in df.columns:
        df["post_holiday_demand_boost"] = (
            df["post_holiday_lag_1"] * df["demand_rolling_mean_4"]
        )

    depot_names = sorted(df["depot"].unique())
    depot_enc = {n: i for i, n in enumerate(depot_names)}
    df["depot_enc"] = df["depot"].map(depot_enc)
    return df
