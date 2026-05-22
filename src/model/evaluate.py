import base64
import io
import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.db.db import get_client

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _save_plot_to_db(
    retrain_id: int,
    plot_type: str,
    b64: str,
    depot_id: Optional[int] = None,
) -> None:
    sb = get_client()
    sb.table("tc_model_plots").upsert(
        {
            "retrain_id": retrain_id,
            "plot_type": plot_type,
            "depot_id": depot_id,
            "image_data": f"data:image/png;base64,{b64}",
        },
        on_conflict="retrain_id,plot_type,depot_id",
    ).execute()


def _get_depot_id_map() -> dict[str, int]:
    sb = get_client()
    result = sb.table("tc_depots").select("name,depot_id").execute()
    return {r["name"]: r["depot_id"] for r in result.data}


# ── Individual plot generators ────────────────────────────────

def plot_mape_by_depot(depot_mapes: dict[str, float], retrain_id: int) -> None:
    depots = list(depot_mapes.keys())
    mapes = [depot_mapes[d] for d in depots]
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["tomato" if m > 15 else "steelblue" for m in mapes]
    ax.barh(depots, mapes, color=colors)
    ax.axvline(15, color="red", linestyle="--", label="15% target")
    ax.set_xlabel("MAPE (%)")
    ax.set_title("MAPE by Depot (avg across horizons)")
    ax.legend()
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "mape_by_depot", b64)
    logger.info("[EVAL] mape_by_depot plot saved to DB")


def plot_mape_by_horizon(horizon_mapes: dict[int, float], retrain_id: int) -> None:
    hs = sorted(horizon_mapes.keys())
    mapes = [horizon_mapes[h] for h in hs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hs, mapes, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Forecast Horizon (weeks)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by Forecast Horizon")
    ax.set_xticks(hs)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "mape_by_horizon", b64)
    logger.info("[EVAL] mape_by_horizon plot saved to DB")


def plot_mape_by_season(df: pd.DataFrame, preds_h1: np.ndarray, retrain_id: int) -> None:
    """MAPE split by SW monsoon vs other."""
    valid = df[["is_sw_monsoon", "demand_tonnes"]].copy()
    valid["pred"] = preds_h1[: len(valid)]
    valid = valid.dropna()

    results = {}
    for label, mask in [("SW Monsoon", valid["is_sw_monsoon"] == 1),
                         ("Non-Monsoon", valid["is_sw_monsoon"] == 0)]:
        sub = valid[mask]
        if len(sub):
            mape = np.mean(np.abs((sub["demand_tonnes"] - sub["pred"]) / sub["demand_tonnes"].replace(0, np.nan))) * 100
            results[label] = float(mape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(results.keys(), results.values(), color=["steelblue", "seagreen"])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE: Monsoon vs Non-Monsoon")
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "mape_by_season", b64)
    logger.info("[EVAL] mape_by_season plot saved to DB")


def plot_forecast_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, weeks: pd.Series, retrain_id: int
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weeks, y_true, label="Actual", color="steelblue")
    ax.plot(weeks, y_pred, label="Forecast", color="tomato", linestyle="--")
    ax.set_title("Forecast vs Actual — Last CV Fold (all depots aggregated)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Demand (tonnes)")
    ax.legend()
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "forecast_vs_actual", b64)
    logger.info("[EVAL] forecast_vs_actual plot saved to DB")


def plot_bias_by_depot(bias_map: dict[str, float], retrain_id: int) -> None:
    depots = list(bias_map.keys())
    biases = [bias_map[d] for d in depots]
    colors = ["tomato" if b > 0 else "steelblue" for b in biases]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(depots, biases, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Bias (tonnes) — positive = overforecast")
    ax.set_title("Forecast Bias by Depot")
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "bias_by_depot", b64)
    logger.info("[EVAL] bias_by_depot plot saved to DB")


def plot_feature_importance(model: xgb.XGBRegressor, feature_cols: list[str], retrain_id: int) -> None:
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    names, vals = zip(*sorted_items) if sorted_items else ([], [])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(list(names)[::-1], list(vals)[::-1], color="steelblue")
    ax.set_xlabel("XGBoost Gain")
    ax.set_title("Top 20 Features by Importance (Horizon 1)")
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "feature_importance", b64)
    logger.info("[EVAL] feature_importance plot saved to DB")


def plot_shap_summary(model: xgb.XGBRegressor, X_sample: pd.DataFrame, retrain_id: int) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    b64 = _fig_to_b64(plt.gcf())
    plt.close("all")
    _save_plot_to_db(retrain_id, "shap_summary", b64)
    logger.info("[EVAL] shap_summary plot saved to DB")


def plot_depot_forecast(
    depot_name: str,
    depot_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weeks: pd.Series,
    retrain_id: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weeks, y_true, label="Actual", color="steelblue")
    ax.plot(weeks, y_pred, label="Forecast", color="tomato", linestyle="--")
    ax.set_title(f"{depot_name} — Forecast vs Actual (Last CV Fold)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Demand (tonnes)")
    ax.legend()
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "depot_forecast", b64, depot_id=depot_id)


def plot_retrain_history(retrain_id: int) -> None:
    sb = get_client()
    result = sb.table("tc_retrain_log").select(
        "triggered_at,mape_before,mape_after"
    ).eq("status", "completed").order("triggered_at").execute()
    df = pd.DataFrame(result.data)

    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["triggered_at"], df["mape_after"], marker="o", label="MAPE after", color="steelblue")
    ax.plot(df["triggered_at"], df["mape_before"], marker="x", linestyle="--",
            label="MAPE before", color="tomato", alpha=0.6)
    ax.set_xlabel("Retrain date")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Model MAPE Trend Across Retraining Runs")
    ax.legend()
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    _save_plot_to_db(retrain_id, "retrain_history", b64)
    logger.info("[EVAL] retrain_history plot saved to DB")


# ── Main evaluation entry point ───────────────────────────────

def run_evaluation(training_result: dict, df_full: pd.DataFrame, retrain_id: int, cfg: dict) -> dict:
    """
    Generate all evaluation plots and save to model_plots table.
    Returns evaluation summary.
    """
    horizon_results = training_result["horizon_results"]
    feature_cols = training_result["feature_cols"]
    depot_id_map = _get_depot_id_map()

    # Aggregate depot MAPEs across all horizons
    all_depot_mapes: dict[str, list[float]] = {}
    all_depot_biases: dict[str, list[float]] = {}
    horizon_mapes: dict[int, float] = {}

    for h, res in horizon_results.items():
        horizon_mapes[h] = res["mape"]
        for depot, dm in res["depot_mapes"].items():
            all_depot_mapes.setdefault(depot, []).append(dm)

    avg_depot_mapes = {d: float(np.mean(v)) for d, v in all_depot_mapes.items()}

    # Use horizon=1 model for feature importance, SHAP, forecast vs actual
    h1_res = horizon_results.get(1, list(horizon_results.values())[0])
    h1_model = h1_res["model"]

    # Forecast vs actual on last CV fold (aggregated)
    df_full["week_start"] = pd.to_datetime(df_full["week_start"])
    df_full[f"target_h1"] = df_full.groupby("depot")["demand_tonnes"].shift(-1)
    valid_h1 = df_full.dropna(subset=[f"target_h1"] + feature_cols)
    weeks_sorted = sorted(valid_h1["week_start"].unique())
    last_6_weeks = set(weeks_sorted[-6:])
    last_fold = valid_h1[valid_h1["week_start"].isin(last_6_weeks)]

    Xva = last_fold[feature_cols]
    yva = last_fold["target_h1"]
    weeks_va = last_fold["week_start"]
    preds_va = h1_model.predict(Xva)

    # Aggregate to weekly totals for forecast vs actual plot
    agg_actual = last_fold.groupby("week_start")["target_h1"].sum()
    agg_pred_s = pd.Series(preds_va, index=last_fold.index)
    agg_pred = last_fold[["week_start"]].assign(pred=agg_pred_s.values).groupby("week_start")["pred"].sum()
    common_weeks = sorted(agg_actual.index.intersection(agg_pred.index))

    plot_forecast_vs_actual(
        agg_actual.loc[common_weeks].values,
        agg_pred.loc[common_weeks].values,
        pd.Series(common_weeks),
        retrain_id,
    )

    # Bias per depot
    bias_map = {}
    for depot in last_fold["depot"].unique():
        mask = last_fold["depot"] == depot
        y_t = last_fold.loc[mask, "target_h1"].values
        y_p = preds_va[mask.values]
        bias_map[depot] = float(np.mean(y_p - y_t))

    # Global plots
    plot_mape_by_depot(avg_depot_mapes, retrain_id)
    plot_mape_by_horizon(horizon_mapes, retrain_id)
    plot_mape_by_season(last_fold, preds_va, retrain_id)
    plot_bias_by_depot(bias_map, retrain_id)
    plot_feature_importance(h1_model, feature_cols, retrain_id)

    # SHAP on 500-row sample
    sample = valid_h1[feature_cols].sample(min(500, len(valid_h1)), random_state=42)
    try:
        plot_shap_summary(h1_model, sample, retrain_id)
    except Exception as e:
        logger.warning("[EVAL] SHAP plot failed (non-fatal): %s", e)

    plot_retrain_history(retrain_id)

    # Per-depot forecast plots (one per depot)
    for depot in last_fold["depot"].unique():
        mask = last_fold["depot"] == depot
        y_t = last_fold.loc[mask, "target_h1"].values
        y_p = preds_va[mask.values]
        w = last_fold.loc[mask, "week_start"]
        d_id = depot_id_map.get(depot)
        plot_depot_forecast(depot, d_id, y_t, y_p, w, retrain_id)

    overall_mape = float(np.mean(list(horizon_mapes.values())))
    logger.info("[EVAL] Evaluation complete. Overall MAPE: %.2f%%", overall_mape)
    logger.info("[EVAL] Depot MAPEs: %s",
                {d: f"{v:.1f}%" for d, v in sorted(avg_depot_mapes.items())})

    return {
        "overall_mape": overall_mape,
        "horizon_mapes": horizon_mapes,
        "depot_mapes": avg_depot_mapes,
    }
