"""
pipeline.py — Single entry point for the Tokyo Cement Demand Forecasting System.

Usage:
    python pipeline.py --mode setup         # First-time setup
    python pipeline.py --mode update        # Fetch fresh weather/economic data
    python pipeline.py --mode train         # Train (or retrain) the model + push forecasts
    python pipeline.py --mode forecast_all  # Push 6-week forecasts for all 24 depots to DB
    python pipeline.py --mode serve         # Start the FastAPI server
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
    """
    Fetch fresh weather from the API and update tc_demand_panel directly.

    Two passes per depot:
      Pass 1 — rows with NULL weather (any date): fetch the appropriate tier
               based on the week date and UPDATE those rows in-place.
      Pass 2 — weeks beyond the current DB ceiling: INSERT new rows with
               weather + calendar + economics already populated.

    No intermediate CSV files are used for the update path.
    """
    logger.info("[PIPELINE] ══ MODE: update ══")

    import time
    import pandas as pd
    from datetime import date, timedelta
    from src.db.db import get_client
    from src.ingestion.weather_ingest import _fetch_open_meteo, _agg_hourly_to_weekly
    from src.ingestion.economic_ingest import fetch_worldbank
    from src.ingestion.calendar_build import build_calendar

    sb = get_client()
    lag_days = int(cfg["weather"]["tier2_lag_days"])
    today = date.today()
    tier2_end = today - timedelta(days=lag_days)

    # ── Step 1: Latest week and depot map ────────────────────────
    logger.info("[PIPELINE] Step 1/4 — Query DB state")
    result = sb.table("tc_demand_panel").select("week_start").order(
        "week_start", desc=True
    ).limit(1).execute()
    latest_week = pd.Timestamp(result.data[0]["week_start"]) if result.data else None
    if not latest_week:
        logger.error("[PIPELINE] tc_demand_panel is empty. Run --mode setup first.")
        sys.exit(1)
    logger.info("[PIPELINE] Latest week in DB: %s", latest_week.date())

    depot_result = sb.table("tc_depots").select("name,depot_id").execute()
    depot_id_map = {r["name"]: r["depot_id"] for r in depot_result.data}

    # ── Step 2: Refresh economics + calendar ────────────────────
    logger.info("[PIPELINE] Step 2/4 — Refresh World Bank economics")
    econ_path = os.path.join(cfg["paths"]["raw_economic"], "worldbank_lka.csv")
    if os.path.exists(econ_path):
        os.remove(econ_path)
    econ = fetch_worldbank(cfg)

    cal = build_calendar(cfg)

    weather_cols = ["precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean"]
    cal_cols = ["is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
                "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
                "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
                "week_of_year", "month"]
    econ_cols = ["gdp_lka", "lending_rate", "govt_consumption"]

    # ── Step 3: Per-depot weather fetch + DB update ──────────────
    logger.info("[PIPELINE] Step 3/4 — Fetch weather from API and update DB rows")
    total_updated = 0
    total_inserted = 0

    for depot in cfg["depots"]:
        name = depot["name"]
        depot_id = depot_id_map.get(name)
        if not depot_id:
            continue

        logger.info("[PIPELINE] Processing depot: %s", name)

        # Find weeks for this depot that need weather (NULL precip_sum)
        null_result = sb.table("tc_demand_panel").select("week_start").eq(
            "depot_id", depot_id
        ).is_("precip_sum", "null").order("week_start").execute()
        null_weeks = [pd.Timestamp(r["week_start"]) for r in null_result.data]

        if not null_weeks:
            logger.info("[PIPELINE] %s — no NULL weather rows, checking for new weeks only", name)

        # Determine full date range to fetch: earliest null week to today
        fetch_start = null_weeks[0].date() if null_weeks else (latest_week + pd.Timedelta(weeks=1)).date()
        fetch_start = min(fetch_start, (latest_week + pd.Timedelta(weeks=1)).date())

        try:
            base = {
                "latitude": depot["lat"],
                "longitude": depot["lon"],
                "hourly": cfg["weather"]["hourly_vars"],
                "timezone": cfg["weather"]["timezone"],
                "format": "csv",
            }
            frames = []

            # Tier 2: historical forecast archive for dates before the rolling window
            if pd.Timestamp(fetch_start) < pd.Timestamp(tier2_end):
                p2 = {**base, "start_date": fetch_start.isoformat(),
                      "end_date": tier2_end.isoformat()}
                try:
                    df2 = _fetch_open_meteo(cfg["weather"]["urls"]["tier2"], p2)
                    frames.append(_agg_hourly_to_weekly(df2))
                    logger.info("[PIPELINE] %s — Tier2 fetched (%s → %s)",
                                name, fetch_start, tier2_end)
                except Exception as e:
                    logger.warning("[PIPELINE] %s — Tier2 failed (non-fatal): %s", name, e)

            # Tier 3: current rolling window
            p3 = {**base, "past_days": lag_days, "forecast_days": 0}
            try:
                df3 = _fetch_open_meteo(cfg["weather"]["urls"]["current"], p3)
                frames.append(_agg_hourly_to_weekly(df3))
                logger.info("[PIPELINE] %s — Tier3 fetched", name)
            except Exception as e:
                logger.warning("[PIPELINE] %s — Tier3 failed (non-fatal): %s", name, e)

            if not frames:
                logger.warning("[PIPELINE] %s — no weather data fetched, skipping", name)
                time.sleep(3)
                continue

            weather_df = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["week_start"])
                .sort_values("week_start")
                .reset_index(drop=True)
            )
            weather_df = weather_df[weather_df["week_start"] >= pd.Timestamp(fetch_start)]

        except Exception as e:
            logger.error("[PIPELINE] %s — weather fetch failed: %s", name, e)
            time.sleep(3)
            continue

        # Build update/insert records
        null_week_set = {w.date().isoformat() for w in null_weeks}
        cutoff = latest_week.date().isoformat()

        updates = []
        inserts = []

        for _, wrow in weather_df.iterrows():
            week_str = wrow["week_start"].strftime("%Y-%m-%d")
            w_vals = {}
            for col in weather_cols:
                v = wrow.get(col)
                w_vals[col] = float(v) if pd.notna(v) else None

            if week_str in null_week_set:
                # Existing row with NULL weather — UPDATE only weather cols
                updates.append({"week_start": week_str, **w_vals})

            elif week_str > cutoff:
                # New week beyond DB ceiling — INSERT with all context
                cal_row = cal[cal["week_start"] == wrow["week_start"]]
                econ_row = econ[econ["week_start"] == wrow["week_start"]]
                rec = {
                    "depot_id": depot_id,
                    "week_start": week_str,
                    "data_source": "augmented",
                    **w_vals,
                }
                if not cal_row.empty:
                    for col in cal_cols:
                        if col in cal_row.columns:
                            v = cal_row[col].iloc[0]
                            rec[col] = int(v) if col not in ("week_of_year", "month") else int(v)
                if not econ_row.empty:
                    for col in econ_cols:
                        if col in econ_row.columns:
                            v = econ_row[col].iloc[0]
                            rec[col] = float(v) if pd.notna(v) else None
                inserts.append(rec)

        # Apply updates (PATCH weather into existing rows)
        for rec in updates:
            sb.table("tc_demand_panel").update(
                {k: v for k, v in rec.items() if k != "week_start"}
            ).eq("depot_id", depot_id).eq("week_start", rec["week_start"]).execute()
        total_updated += len(updates)

        # Apply inserts (new weeks)
        if inserts:
            batch_size = 200
            for i in range(0, len(inserts), batch_size):
                sb.table("tc_demand_panel").upsert(
                    inserts[i: i + batch_size],
                    on_conflict="depot_id,week_start",
                ).execute()
            total_inserted += len(inserts)

        logger.info("[PIPELINE] %s — %d weather rows updated, %d new rows inserted",
                    name, len(updates), len(inserts))
        time.sleep(3)  # respect Open-Meteo rate limit (3 req/s free tier)

    # ── Step 4: Summary ──────────────────────────────────────────
    logger.info(
        "[PIPELINE] ✓ Update complete\n"
        "  Weather rows updated: %d\n"
        "  New weeks inserted:   %d\n"
        "  Depots processed:     %d",
        total_updated,
        total_inserted,
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

    # Step 3: Evaluate + save plots (non-fatal — requires tc_model_plots table)
    logger.info("[PIPELINE] Step 3/5 — Evaluate model and save plots")
    try:
        from src.model.evaluate import run_evaluation
        run_evaluation(result, df, retrain_id, cfg)
    except Exception as e:
        logger.warning("[PIPELINE] Evaluation skipped (non-fatal): %s", e)

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

    # Step 6: Push fresh forecasts for all depots to tc_forecasts
    logger.info("[PIPELINE] Step 6/6 — Push 6-week forecasts for all depots to DB")
    run_forecast_all(cfg)


# ════════════════════════════════════════════════════════════
# MODE: forecast_all
# ════════════════════════════════════════════════════════════

def run_forecast_all(cfg: dict) -> None:
    """Generate a 6-week forecast for every depot and upsert into tc_forecasts."""
    import pandas as pd
    from datetime import date
    from src.db.db import get_client
    from src.model.predict import load_models, forecast_depot

    logger.info("[PIPELINE] ══ MODE: forecast_all ══")
    load_models(cfg)

    sb = get_client()
    depot_result = sb.table("tc_depots").select("depot_id,name").order("name").execute()
    depots = depot_result.data
    today = date.today()
    total = 0

    for depot in depots:
        depot_id = depot["depot_id"]
        depot_name = depot["name"]

        panel_result = sb.table("tc_demand_panel").select("*").eq(
            "depot_id", depot_id
        ).order("week_start", desc=True).limit(52).execute()

        df = pd.DataFrame(panel_result.data)
        if df.empty:
            logger.warning("[PIPELINE] forecast_all: no panel data for depot '%s', skipping", depot_name)
            continue

        df["depot"] = depot_name
        df = df.sort_values("week_start").reset_index(drop=True)

        try:
            forecasts = forecast_depot(depot_name, today, df, cfg)
        except Exception as e:
            logger.warning("[PIPELINE] forecast_all: depot '%s' failed: %s", depot_name, e)
            continue

        for fc in forecasts:
            sb.table("tc_forecasts").upsert({
                "depot_id": depot_id,
                "as_of_date": today.isoformat(),
                "horizon_weeks": fc["horizon"],
                "forecast_week": fc["forecast_week"].isoformat(),
                "demand_forecast": fc["demand_tonnes"],
            }, on_conflict="depot_id,forecast_week").execute()

        total += 1
        logger.info("[PIPELINE] forecast_all: %s — %d horizons stored", depot_name, len(forecasts))

    logger.info("[PIPELINE] ✓ forecast_all complete: %d/%d depots", total, len(depots))


# ════════════════════════════════════════════════════════════
# MODE: serve
# ════════════════════════════════════════════════════════════

def run_serve(cfg: dict) -> None:
    logger.info("[PIPELINE] ══ MODE: serve ══")
    import uvicorn

    host = os.environ.get("API_HOST", cfg["api"]["host"])
    # Railway injects PORT; fall back to API_PORT, then config default
    port = int(os.environ.get("PORT", os.environ.get("API_PORT", cfg["api"]["port"])))

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
        choices=["setup", "update", "train", "forecast_all", "serve"],
        help="Pipeline mode to run",
    )
    args = parser.parse_args()

    _require_env("DATABASE_URL")

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
        "forecast_all": run_forecast_all,
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
