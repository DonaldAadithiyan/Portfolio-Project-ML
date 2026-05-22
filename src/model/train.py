import logging
import os
import warnings
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = []  # populated at runtime


def _get_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return ordered list of feature columns (excluding target and metadata)."""
    exclude = {
        "week_start", "depot", "demand_tonnes", "data_source",
        "sales_tonnes", "production_tonnes",
    }
    cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("Unnamed")
        and df[c].notna().any()  # drop columns that are entirely NaN
    ]
    return cols


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _rolling_cv_splits(df: pd.DataFrame, cfg: dict) -> list[tuple]:
    """
    Generate rolling-window CV splits on the sorted week index.
    Each split: (train_idx, val_idx)
    """
    m = cfg["model"]
    train_w = int(m["cv_train_weeks"])
    val_w = int(m["cv_val_weeks"])
    step = int(m["cv_step_weeks"])

    weeks = sorted(df["week_start"].unique())
    n = len(weeks)
    splits = []
    start = 0
    while start + train_w + val_w <= n:
        train_weeks = set(weeks[start: start + train_w])
        val_weeks = set(weeks[start + train_w: start + train_w + val_w])
        tr_idx = df[df["week_start"].isin(train_weeks)].index.tolist()
        va_idx = df[df["week_start"].isin(val_weeks)].index.tolist()
        if tr_idx and va_idx:
            splits.append((tr_idx, va_idx))
        start += step

    min_folds = int(m["cv_min_folds"])
    if len(splits) < min_folds:
        raise RuntimeError(
            f"[TRAIN] Only {len(splits)} CV folds found, need {min_folds}. "
            "Check data length or cv parameters."
        )
    return splits


def _tune_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict,
    horizon: int,
) -> dict:
    """Run Optuna to find best XGBoost hyperparams on a single (train, val) split."""
    xgb_cfg = cfg["model"]["xgb"]
    n_trials = int(cfg["model"]["optuna_trials"])

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", *xgb_cfg["n_estimators"]),
            "max_depth":        trial.suggest_int("max_depth", *xgb_cfg["max_depth"]),
            "learning_rate":    trial.suggest_float("learning_rate", *xgb_cfg["learning_rate"], log=True),
            "subsample":        trial.suggest_float("subsample", *xgb_cfg["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *xgb_cfg["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *xgb_cfg["min_child_weight"]),
            "tree_method":      "hist",
            "random_state":     42,
            "verbosity":        0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return _mape(y_val.values, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("[TRAIN] horizon=%d best MAPE=%.2f%% params=%s",
                horizon, study.best_value, study.best_params)
    return study.best_params


def _load_training_data_from_db(cfg: dict) -> pd.DataFrame:
    """Pull full demand_panel from DB and rebuild features."""
    from src.features.build_features import rebuild_lag_features_for_depots
    logger.info("[TRAIN] Loading training data from demand_panel DB table")
    df = rebuild_lag_features_for_depots([], cfg)  # empty list = all depots
    logger.info("[TRAIN] Loaded %d rows from DB", len(df))
    return df


def _get_current_production_mape(cfg: dict) -> float | None:
    """Return MAPE of the current Production model from MLflow registry, or None."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(
            cfg["model"]["registry_name"], stages=["Production"]
        )
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return float(run.data.metrics.get("mape_val_avg", run.data.metrics.get("mape_val", 999)))
    except Exception as e:
        logger.warning("[TRAIN] Could not fetch production MAPE: %s", e)
        return None


def train_all_horizons(cfg: dict, retrain_id: int | None = None) -> dict:
    """
    Train 6 XGBoost models (one per horizon) using rolling-window CV.
    Logs to MLflow, registers best models.
    Returns summary dict.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    tracking_user = os.getenv("MLFLOW_TRACKING_USERNAME")
    tracking_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if tracking_user and tracking_pass:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_pass

    # Load data
    df = _load_training_data_from_db(cfg)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["depot", "week_start"]).reset_index(drop=True)

    feature_cols = _get_feature_cols(df, cfg)
    logger.info("[TRAIN] Feature columns (%d): %s", len(feature_cols), feature_cols)

    horizons = cfg["model"]["horizons"]
    cv_splits = _rolling_cv_splits(df, cfg)
    logger.info("[TRAIN] CV splits: %d", len(cv_splits))

    current_prod_mape = _get_current_production_mape(cfg)
    logger.info("[TRAIN] Current production MAPE: %s", current_prod_mape)

    experiment_name = cfg["model"]["registry_name"]
    mlflow.set_experiment(experiment_name)

    horizon_results = {}
    all_mapes = []

    for horizon in horizons:
        logger.info("[TRAIN] ─── Horizon %d ───", horizon)

        # Build target: shift demand_tonnes back by `horizon` weeks per depot
        df[f"target_h{horizon}"] = df.groupby("depot")["demand_tonnes"].shift(-horizon)
        valid = df.dropna(subset=[f"target_h{horizon}"] + feature_cols)

        fold_mapes = []
        fold_maes = []
        fold_biases = []
        best_params = None

        # Use last two folds for tuning (faster)
        tune_split = cv_splits[-2] if len(cv_splits) >= 2 else cv_splits[-1]
        tr_idx, va_idx = tune_split
        Xtr = valid.loc[[i for i in tr_idx if i in valid.index], feature_cols]
        ytr = valid.loc[[i for i in tr_idx if i in valid.index], f"target_h{horizon}"]
        Xva = valid.loc[[i for i in va_idx if i in valid.index], feature_cols]
        yva = valid.loc[[i for i in va_idx if i in valid.index], f"target_h{horizon}"]

        if len(Xtr) > 10 and len(Xva) > 0:
            best_params = _tune_hyperparams(Xtr, ytr, Xva, yva, cfg, horizon)
        else:
            best_params = {
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            }

        # Full rolling CV evaluation
        for tr_idx, va_idx in cv_splits:
            Xtr = valid.loc[[i for i in tr_idx if i in valid.index], feature_cols]
            ytr = valid.loc[[i for i in tr_idx if i in valid.index], f"target_h{horizon}"]
            Xva = valid.loc[[i for i in va_idx if i in valid.index], feature_cols]
            yva = valid.loc[[i for i in va_idx if i in valid.index], f"target_h{horizon}"]

            if len(Xtr) < 10 or len(Xva) == 0:
                continue

            m = xgb.XGBRegressor(
                **best_params, tree_method="hist", random_state=42, verbosity=0
            )
            m.fit(Xtr, ytr, verbose=False)
            preds = m.predict(Xva)
            fold_mapes.append(_mape(yva.values, preds))
            fold_maes.append(_mae(yva.values, preds))
            fold_biases.append(_bias(yva.values, preds))

        avg_mape = float(np.mean(fold_mapes)) if fold_mapes else 999.0
        avg_mae = float(np.mean(fold_maes)) if fold_maes else 999.0
        avg_bias = float(np.mean(fold_biases)) if fold_biases else 0.0
        all_mapes.append(avg_mape)

        # Train final model on all data (drop last 6 weeks)
        all_weeks = sorted(valid["week_start"].unique())
        cutoff = all_weeks[-int(cfg["model"]["cv_val_weeks"])] if len(all_weeks) > int(cfg["model"]["cv_val_weeks"]) else all_weeks[0]
        train_final = valid[valid["week_start"] < cutoff]
        Xfull = train_final[feature_cols]
        yfull = train_final[f"target_h{horizon}"]

        final_model = xgb.XGBRegressor(
            **best_params, tree_method="hist", random_state=42, verbosity=0
        )
        final_model.fit(Xfull, yfull, verbose=False)

        # Per-depot MAPE on last CV fold
        last_tr, last_va = cv_splits[-1]
        Xva_last = valid.loc[[i for i in last_va if i in valid.index], feature_cols]
        yva_last = valid.loc[[i for i in last_va if i in valid.index], f"target_h{horizon}"]
        depot_last = valid.loc[[i for i in last_va if i in valid.index], "depot"]
        preds_last = final_model.predict(Xva_last)

        depot_mapes = {}
        for depot in depot_last.unique():
            mask = depot_last == depot
            dm = _mape(yva_last[mask].values, preds_last[mask])
            depot_mapes[depot] = dm

        # MLflow run
        with mlflow.start_run(run_name=f"xgb_horizon_{horizon}") as run:
            mlflow.log_params({**best_params, "horizon": horizon,
                               "features": len(feature_cols),
                               "train_window_weeks": cfg["model"]["cv_train_weeks"]})
            if retrain_id is not None:
                mlflow.set_tag("retrain_id", str(retrain_id))
            mlflow.log_metrics({
                "mape_val": avg_mape,
                "mae_val": avg_mae,
                "bias_val": avg_bias,
            })
            for depot, dm in depot_mapes.items():
                safe_name = depot.lower().replace(" ", "_")
                mlflow.log_metric(f"mape_per_depot_{safe_name}", dm)

            mlflow.xgboost.log_model(
                final_model,
                artifact_path=f"model_h{horizon}",
                registered_model_name=f"{cfg['model']['registry_name']}_h{horizon}",
            )
            run_id = run.info.run_id

        horizon_results[horizon] = {
            "mape": avg_mape, "mae": avg_mae, "bias": avg_bias,
            "depot_mapes": depot_mapes, "run_id": run_id,
            "model": final_model, "feature_cols": feature_cols,
        }
        logger.info("[TRAIN] horizon=%d mape=%.2f%% mae=%.1f bias=%.1f",
                    horizon, avg_mape, avg_mae, avg_bias)

    overall_mape = float(np.mean(all_mapes))
    logger.info("[TRAIN] Overall average MAPE across horizons: %.2f%%", overall_mape)

    # ── Promotion decision ────────────────────────────────────
    promoted = False
    if current_prod_mape is None or overall_mape <= current_prod_mape:
        promoted = True
        _promote_models(cfg, horizon_results)
        logger.info("[TRAIN] New models promoted to Production (MAPE %.2f%% vs previous %.2f%%)",
                    overall_mape, current_prod_mape or float("inf"))
    else:
        logger.info("[TRAIN] Models NOT promoted — MAPE %.2f%% worse than production %.2f%%",
                    overall_mape, current_prod_mape)

    return {
        "overall_mape": overall_mape,
        "mape_before": current_prod_mape,
        "promoted": promoted,
        "horizon_results": horizon_results,
        "feature_cols": feature_cols,
    }


def _promote_models(cfg: dict, horizon_results: dict) -> None:
    """Transition each horizon model to Production in MLflow registry."""
    client = MlflowClient()
    registry_name = cfg["model"]["registry_name"]
    for horizon in cfg["model"]["horizons"]:
        model_name = f"{registry_name}_h{horizon}"
        versions = client.get_latest_versions(model_name)
        if not versions:
            continue
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("[TRAIN] Promoted %s v%s to Production", model_name, latest.version)
