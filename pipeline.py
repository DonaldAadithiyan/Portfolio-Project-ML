"""
pipeline.py — Single entry point for the Tokyo Cement Demand Forecasting System.

Usage:
    python pipeline.py --mode setup    # First-time setup
    python pipeline.py --mode update   # Fetch fresh weather/economic data
    python pipeline.py --mode train    # Train (or retrain) the model
    python pipeline.py --mode serve    # Start the FastAPI server
"""

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        logger.error("[PIPELINE] Missing required environment variable: %s", key)
        sys.exit(1)
    return val


# ════════════════════════════════════════════════════════════
# MODE: setup
# ════════════════════════════════════════════════════════════

def run_setup(cfg: dict) -> None:
    logger.info("[PIPELINE] ══ MODE: setup ══")

    # Step 1: Download Kaggle dataset
    logger.info("[PIPELINE] Step 1/9 — Download Kaggle dataset")
    from src.ingestion.kaggle_ingest import download_kaggle_dataset, load_kaggle_csv
    kaggle_path = download_kaggle_dataset(cfg)
    kaggle_df = load_kaggle_csv(kaggle_path)

    # Step 2: Pull weather data (all 3 tiers)
    logger.info("[PIPELINE] Step 2/9 — Pull weather data (ERA5 + Tier2 + Tier3)")
    from src.ingestion.weather_ingest import fetch_all_depots_weather
    fetch_all_depots_weather(cfg, tier="all")

    # Step 3: Pull World Bank economic data
    logger.info("[PIPELINE] Step 3/9 — Pull World Bank economic data")
    from src.ingestion.economic_ingest import fetch_worldbank, fetch_pink_sheet, load_cbsl_pmi
    worldbank_df = fetch_worldbank(cfg)
    fetch_pink_sheet(cfg)
    pmi_df = load_cbsl_pmi(cfg)

    # Save PMI as a separate weekly CSV for the feature join
    if not pmi_df.empty and "cbsl_pmi_construction" in pmi_df.columns:
        import pandas as pd
        pmi_path = os.path.join(cfg["paths"]["raw_economic"], "cbsl_pmi_weekly.csv")
        pmi_df.to_csv(pmi_path, index=False)
        logger.info("[PIPELINE] CBSL PMI saved: %s", pmi_path)

    # Step 4: Build Sri Lanka calendar table
    logger.info("[PIPELINE] Step 4/9 — Build Sri Lanka calendar table")
    from src.ingestion.calendar_build import build_calendar
    build_calendar(cfg)

    # Step 5: Augmentation pipeline
    logger.info("[PIPELINE] Step 5/9 — Augmentation pipeline")

    from src.augmentation.replace_economics import replace_economics
    df_econ = replace_economics(kaggle_df, worldbank_df, cfg)

    from src.augmentation.scale_to_lka import scale_to_lka
    df_scaled = scale_to_lka(df_econ, cfg)

    from src.augmentation.disaggregate_weekly import disaggregate_weekly
    df_weekly = disaggregate_weekly(df_scaled, cfg)

    from src.augmentation.seasonal_override import seasonal_override
    df_seasadj = seasonal_override(df_weekly, cfg)

    from src.augmentation.split_to_depots import split_to_depots
    df_depot = split_to_depots(df_seasadj, cfg)

    # Step 6: Build features and assemble final panel
    logger.info("[PIPELINE] Step 6/9 — Build features and assemble final panel")
    from src.features.build_features import build_features
    panel = build_features(cfg)

    # Step 7: Verify database schema exists
    logger.info("[PIPELINE] Step 7/9 — Verify database schema")
    from src.db.db import check_schema
    if not check_schema():
        logger.error(
            "[PIPELINE] tc_depots table not found in Supabase.\n"
            "  → Open the Supabase SQL Editor and run: src/db/schema.sql\n"
            "  → Then re-run: python pipeline.py --mode setup"
        )
        sys.exit(1)
    logger.info("[PIPELINE] Schema OK")

    # Step 8: Seed depots and demand_panel
    logger.info("[PIPELINE] Step 8/9 — Seed depots and demand_panel")
    from src.db.seed import seed_depots, seed_demand_panel
    depot_map = seed_depots(cfg["depots"])

    processed_path = os.path.join(cfg["paths"]["processed"], "panel_modelling.csv")
    rows_written = seed_demand_panel(processed_path, depot_map)

    # Step 9: Summary
    logger.info("[PIPELINE] Step 9/9 — Summary")
    import pandas as pd
    df_summary = pd.read_csv(processed_path, parse_dates=["week_start"])
    logger.info(
        "[PIPELINE] ✓ Setup complete\n"
        "  Rows written:   %d\n"
        "  Date range:     %s – %s\n"
        "  Depots seeded:  %d",
        rows_written,
        df_summary["week_start"].min().date(),
        df_summary["week_start"].max().date(),
        len(depot_map),
    )


# ════════════════════════════════════════════════════════════
# MODE: update
# ════════════════════════════════════════════════════════════

def run_update(cfg: dict) -> None:
    logger.info("[PIPELINE] ══ MODE: update ══")

    import pandas as pd
    from src.db.db import get_client

    # Step 1: Find latest week in tc_demand_panel
    logger.info("[PIPELINE] Step 1/5 — Query latest week in tc_demand_panel")
    sb = get_client()
    result = sb.table("tc_demand_panel").select("week_start").order(
        "week_start", desc=True
    ).limit(1).execute()
    latest_week = result.data[0]["week_start"] if result.data else None

    if not latest_week:
        logger.error("[PIPELINE] tc_demand_panel is empty. Run --mode setup first.")
        sys.exit(1)
    logger.info("[PIPELINE] Latest week in DB: %s", latest_week)

    # Step 2: Pull Tier 3 weather for all depots (current rolling window)
    logger.info("[PIPELINE] Step 2/5 — Pull fresh weather data (Tier 3 only)")
    from src.ingestion.weather_ingest import fetch_all_depots_weather
    fetch_all_depots_weather(cfg, tier="tier3")

    # Step 3: Pull any new World Bank data
    logger.info("[PIPELINE] Step 3/5 — Pull updated World Bank economic data")
    from src.ingestion.economic_ingest import fetch_worldbank
    econ_path = os.path.join(cfg["paths"]["raw_economic"], "worldbank_lka.csv")
    if os.path.exists(econ_path):
        os.remove(econ_path)
    fetch_worldbank(cfg)

    # Step 4: Append new rows to tc_demand_panel
    logger.info("[PIPELINE] Step 4/5 — Append new weekly rows to tc_demand_panel")
    new_weeks_added = 0
    try:
        from src.ingestion.calendar_build import build_calendar
        cal = build_calendar(cfg)
        econ = pd.read_csv(econ_path, parse_dates=["week_start"])

        weather_dir = cfg["paths"]["raw_weather"]
        first_depot = cfg["depots"][0]
        fname = first_depot["name"].lower().replace(" ", "_") + ".csv"
        w_df = pd.read_csv(os.path.join(weather_dir, fname), parse_dates=["week_start"])
        new_weather_weeks = w_df[w_df["week_start"] > pd.Timestamp(latest_week)]["week_start"]

        depot_result = sb.table("tc_depots").select("name,depot_id").execute()
        depot_map = {r["name"]: r["depot_id"] for r in depot_result.data}

        new_rows = []
        for new_week in sorted(new_weather_weeks):
            cal_row = cal[cal["week_start"] == new_week]
            econ_row = econ[econ["week_start"] == new_week]

            for depot in cfg["depots"]:
                depot_id = depot_map.get(depot["name"])
                if not depot_id:
                    continue

                wf = os.path.join(weather_dir, depot["name"].lower().replace(" ", "_") + ".csv")
                w_df2 = pd.read_csv(wf, parse_dates=["week_start"])
                week_weather = w_df2[w_df2["week_start"] == new_week]

                row = {
                    "depot_id": depot_id,
                    "week_start": new_week.date().isoformat(),
                    "data_source": "augmented",
                }
                if not week_weather.empty:
                    for col in ["precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean"]:
                        if col in week_weather.columns:
                            row[col] = float(week_weather[col].iloc[0])
                if not cal_row.empty:
                    for col in ["is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
                                "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
                                "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter"]:
                        if col in cal_row.columns:
                            row[col] = int(cal_row[col].iloc[0])
                if not econ_row.empty:
                    for col in ["gdp_lka", "lending_rate", "govt_consumption"]:
                        if col in econ_row.columns:
                            row[col] = float(econ_row[col].iloc[0])
                new_rows.append(row)

        if new_rows:
            batch_size = 200
            for i in range(0, len(new_rows), batch_size):
                sb.table("tc_demand_panel").upsert(
                    new_rows[i : i + batch_size],
                    on_conflict="depot_id,week_start",
                ).execute()
            new_weeks_added = len(set(r["week_start"] for r in new_rows))

    except Exception as e:
        logger.error("[PIPELINE] update step 4 failed: %s", e)
        raise

    # Step 5: Summary
    logger.info(
        "[PIPELINE] ✓ Update complete\n"
        "  Weeks added:   %d\n"
        "  Depots updated: %d",
        new_weeks_added,
        len(cfg["depots"]),
    )


# ════════════════════════════════════════════════════════════
# MODE: train
# ════════════════════════════════════════════════════════════

def run_train(cfg: dict) -> None:
    logger.info("[PIPELINE] ══ MODE: train ══")

    # Step 1: Pull training data from DB
    logger.info("[PIPELINE] Step 1/5 — Pull training data from DB")
    from src.features.build_features import rebuild_lag_features_for_depots
    df = rebuild_lag_features_for_depots([], cfg)
    logger.info("[PIPELINE] Training data: %d rows", len(df))

    # Step 2–4: Rolling CV + XGBoost training + MLflow logging
    logger.info("[PIPELINE] Step 2/5 — Run rolling-window CV and train XGBoost models")
    from src.model.train import train_all_horizons
    from src.db.db import get_client

    sb = get_client()
    log_row = sb.table("tc_retrain_log").insert({
        "triggered_by": "pipeline_cli",
        "trigger_reason": "--mode train",
        "status": "running",
    }).execute()
    retrain_id = log_row.data[0]["id"]

    result = train_all_horizons(cfg, retrain_id=retrain_id)

    # Step 3: Evaluate + save plots
    logger.info("[PIPELINE] Step 3/5 — Evaluate model and save plots")
    from src.model.evaluate import run_evaluation
    run_evaluation(result, df, retrain_id, cfg)

    # Step 4: Write retrain_log
    logger.info("[PIPELINE] Step 4/5 — Update retrain_log")
    latest_result = sb.table("tc_demand_panel").select("week_start").order(
        "week_start", desc=True
    ).limit(1).execute()
    latest_week = latest_result.data[0]["week_start"] if latest_result.data else None
    sb.table("tc_retrain_log").update({
        "status": "completed",
        "mape_after": result["overall_mape"],
        "promoted": result["promoted"],
        "training_data_up_to": latest_week,
    }).eq("id", retrain_id).execute()

    # Step 5: Print summary
    logger.info("[PIPELINE] Step 5/5 — Summary")
    mape_before = result.get("mape_before")
    mape_after = result["overall_mape"]
    promoted = result["promoted"]
    print(
        f"\n[TRAIN] Current production model: version {retrain_id} | "
        f"trained {__import__('datetime').date.today()} | "
        f"MAPE {mape_after:.1f}% | promoted: {'yes' if promoted else 'no'}"
    )
    if mape_before:
        print(f"[TRAIN] Previous MAPE: {mape_before:.1f}%")


# ════════════════════════════════════════════════════════════
# MODE: serve
# ════════════════════════════════════════════════════════════

def run_serve(cfg: dict) -> None:
    logger.info("[PIPELINE] ══ MODE: serve ══")
    import uvicorn

    host = os.environ.get("API_HOST", cfg["api"]["host"])
    port = int(os.environ.get("API_PORT", cfg["api"]["port"]))

    logger.info("[PIPELINE] Starting FastAPI on %s:%d", host, port)
    logger.info("[PIPELINE] Endpoints:")
    logger.info("  GET  /depots")
    logger.info("  POST /forecast")
    logger.info("  GET  /forecasts/{depot}")
    logger.info("  POST /stock")
    logger.info("  GET  /stock/{depot}")
    logger.info("  GET  /purchase-orders/{depot}")
    logger.info("  PATCH /purchase-orders/{po_id}")
    logger.info("  GET  /alerts/{depot}")
    logger.info("  PATCH /alerts/{alert_id}/resolve")
    logger.info("  GET  /dashboard/{depot}")
    logger.info("  POST /sales")
    logger.info("  PUT  /sales/{depot}/{week_start}")
    logger.info("  GET  /sales/{depot}")
    logger.info("  POST /retrain")
    logger.info("  GET  /retrain/status/{retrain_id}")
    logger.info("  GET  /retrain/history")
    logger.info("  GET  /plots/latest")
    logger.info("  GET  /plots/depot/{depot}")
    logger.info("  GET  /plots/{retrain_id}")

    uvicorn.run("src.serve.app:app", host=host, port=port, reload=False)


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Tokyo Cement Demand Forecasting Pipeline"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["setup", "update", "train", "serve"],
        help="Pipeline mode to run",
    )
    args = parser.parse_args()

    _require_env("SUPABASE_URL")
    _require_env("SUPABASE_KEY")

    cfg = load_config()

    # Create output directories (idempotent)
    for path_key in ["raw_kaggle", "raw_weather", "raw_economic", "raw_calendar",
                     "interim", "processed", "results"]:
        path = cfg["paths"].get(path_key)
        if path:
            os.makedirs(path, exist_ok=True)

    mode_fns = {
        "setup": run_setup,
        "update": run_update,
        "train": run_train,
        "serve": run_serve,
    }

    try:
        mode_fns[args.mode](cfg)
    except KeyboardInterrupt:
        logger.info("[PIPELINE] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("[PIPELINE] ✗ Mode '%s' failed at step: %s", args.mode, e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
