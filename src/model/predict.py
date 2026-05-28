import logging
import os
from datetime import date, timedelta

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_models: dict[int, mlflow.pyfunc.PyFuncModel] = {}


def load_models(cfg: dict) -> None:
    """Load all 6 horizon models from MLflow Production registry."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    registry = cfg["model"]["registry_name"]
    global _models
    _models = {}
    for h in cfg["model"]["horizons"]:
        model_name = f"{registry}_h{h}"
        uri = f"models:/{model_name}/Production"
        try:
            _models[h] = mlflow.xgboost.load_model(uri)
            logger.info("[PREDICT] Loaded model horizon=%d from %s", h, uri)
        except Exception as e:
            logger.error("[PREDICT] Failed to load model horizon=%d: %s", h, e)
            raise


def _build_feature_row(
    depot_name: str,
    as_of_date: date,
    recent_panel: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Construct a single feature row for inference from recent demand_panel rows.
    recent_panel: last 52 rows for this depot, sorted ascending by week_start.
    """
    # We need to assemble the same feature vector the model was trained on.
    # The lag features are computed from the tail of recent_panel.
    panel = recent_panel.sort_values("week_start").copy()
    panel["week_start"] = pd.to_datetime(panel["week_start"])
    panel["demand_tonnes"] = pd.to_numeric(panel["demand_tonnes"], errors="coerce")

    # Last row is the most recent week — as_of_date context
    last = panel.iloc[-1]

    lag_weeks = cfg["features"]["lag_weeks"]
    rolling_windows = cfg["features"]["rolling_windows"]

    row = {}

    # Lag features
    demands = panel["demand_tonnes"].values
    for lag in lag_weeks:
        row[f"demand_lag_{lag}"] = float(demands[-lag]) if len(demands) >= lag else np.nan

    # Rolling features
    for w in rolling_windows:
        window_vals = demands[-w:] if len(demands) >= w else demands
        row[f"demand_rolling_mean_{w}"] = float(np.mean(window_vals))

    std_window = demands[-4:] if len(demands) >= 4 else demands
    row["demand_rolling_std_4"] = float(np.std(std_window)) if len(std_window) > 1 else 0.0

    # Weather/economic/calendar from last row (same-week context)
    carry_cols = [
        "precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean",
        "gdp_lka", "lending_rate", "govt_consumption",
        "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
        "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
        "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
        "week_of_year", "month",
    ]
    for col in carry_cols:
        if col in last.index:
            try:
                row[col] = float(last[col]) if last[col] is not None else np.nan
            except (TypeError, ValueError):
                row[col] = np.nan

    # Interaction features
    precip = row.get("precip_sum", 0) or 0
    monsoon = row.get("is_sw_monsoon", 0) or 0
    row["precip_x_monsoon"] = precip * monsoon

    ph_lag1 = row.get("post_holiday_lag_1", 0) or 0
    roll_mean4 = row.get("demand_rolling_mean_4", 0) or 0
    row["post_holiday_demand_boost"] = ph_lag1 * roll_mean4

    # Depot encoding (same as training — alphabetical order)
    from src.db.db import get_client
    result = get_client().table("tc_depots").select("name").order("name").execute()
    depot_names = [r["name"] for r in result.data]

    depot_enc = {n: i for i, n in enumerate(depot_names)}
    row["depot_enc"] = depot_enc.get(depot_name, -1)

    return pd.DataFrame([row])


def forecast_depot(
    depot_name: str,
    as_of_date: date,
    recent_panel: pd.DataFrame,
    cfg: dict,
) -> list[dict]:
    """
    Run all 6 horizon models for a depot and return list of forecast dicts.
    """
    if not _models:
        raise RuntimeError("[PREDICT] Models not loaded. Call load_models() first.")

    X = _build_feature_row(depot_name, as_of_date, recent_panel, cfg)

    forecasts = []
    for h in cfg["model"]["horizons"]:
        model = _models[h]
        try:
            feat_names = list(model.feature_names_in_)
            pred = float(model.predict(X[feat_names])[0])
        except Exception as e:
            logger.warning("[PREDICT] horizon=%d predict failed: %s", h, e)
            pred = 0.0

        forecast_week = as_of_date + timedelta(weeks=h)
        forecasts.append({
            "horizon": h,
            "forecast_week": forecast_week,
            "demand_tonnes": round(pred, 2),
        })

    return forecasts
