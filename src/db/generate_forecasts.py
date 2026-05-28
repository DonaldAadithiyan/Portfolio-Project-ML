"""
generate_forecasts.py — Generate fresh 6-week forecasts for all depots as of today.

Run after retraining to populate tc_forecasts with current predictions:
    python -m src.db.generate_forecasts
"""

import logging
import os
import yaml
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.model.predict import load_models, forecast_depot
    from src.db.db import get_client

    logger.info("[FORECAST] Loading models from MLflow registry...")
    load_models(cfg)
    logger.info("[FORECAST] Models loaded.")

    sb = get_client()

    # as_of_date = most recent Monday (last complete week in the panel)
    today = date.today()
    as_of_date = today - timedelta(days=today.weekday())
    logger.info("[FORECAST] Generating forecasts as_of_date = %s", as_of_date)

    depots = sb.table("tc_depots").select("depot_id,name").order("name").execute().data
    logger.info("[FORECAST] %d depots found", len(depots))

    success = 0
    failed = []

    for depot_row in depots:
        depot_id = depot_row["depot_id"]
        depot_name = depot_row["name"]

        # Fetch last 52 weeks of panel data for this depot
        panel_result = sb.table("tc_demand_panel").select("*").eq(
            "depot_id", depot_id
        ).order("week_start", desc=True).limit(52).execute()

        if not panel_result.data:
            logger.warning("[FORECAST] No panel data for %s — skipping", depot_name)
            failed.append(depot_name)
            continue

        import pandas as pd
        panel = pd.DataFrame(panel_result.data)
        panel["depot"] = depot_name
        panel = panel.sort_values("week_start")

        try:
            forecasts = forecast_depot(depot_name, as_of_date, panel, cfg)

            for fc in forecasts:
                sb.table("tc_forecasts").upsert({
                    "depot_id": depot_id,
                    "as_of_date": as_of_date.isoformat(),
                    "horizon_weeks": fc["horizon"],
                    "forecast_week": fc["forecast_week"].isoformat(),
                    "demand_forecast": fc["demand_tonnes"],
                }, on_conflict="depot_id,forecast_week").execute()

            logger.info(
                "[FORECAST] %s — weeks %s to %s",
                depot_name,
                forecasts[0]["forecast_week"].isoformat(),
                forecasts[-1]["forecast_week"].isoformat(),
            )
            success += 1

        except Exception as e:
            logger.error("[FORECAST] Failed for %s: %s", depot_name, e)
            failed.append(depot_name)

    logger.info(
        "[FORECAST] Done. %d/%d depots forecasted. as_of_date=%s",
        success, len(depots), as_of_date,
    )
    if failed:
        logger.warning("[FORECAST] Failed depots: %s", failed)


if __name__ == "__main__":
    main()
