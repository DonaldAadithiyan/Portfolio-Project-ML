"""
backfill_weather.py — One-time script to populate weather columns for the
augmented rows inserted into tc_demand_panel (2022-12-05 to today).

Fetches Tier 2 (historical forecast archive: 2022-12-01 to 92 days ago)
plus Tier 3 (already in CSV files: last 92 days) for every depot, then
UPSERTs the weather columns into tc_demand_panel without touching
demand_tonnes or data_source.

Run once after update_and_augment.sql:
    python -m src.db.backfill_weather
"""

import logging
import os
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,rain,cloud_cover"


def _fetch_open_meteo(url: str, params: dict, retries: int = 5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                wait = 20 * (attempt + 1)
                logger.warning("[WEATHER] 429 rate-limit — waiting %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            lines = r.text.splitlines()
            header_idx = next(
                (i for i, l in enumerate(lines) if l.strip().lower().startswith("time")),
                None,
            )
            if header_idx is None:
                raise ValueError("No 'time' header row in response")
            return pd.read_csv(StringIO("\n".join(lines[header_idx:])))
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (2 ** attempt)
            logger.warning("[WEATHER] Attempt %d failed: %s — retry in %ds", attempt + 1, e, wait)
            time.sleep(wait)


def _agg_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    time_col = next((c for c in df.columns if c.lower() == "time"), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], format="ISO8601", utc=False)
    df = df.rename(columns={time_col: "timestamp"})
    df["week_start"] = df["timestamp"].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    for col in df.columns:
        if col not in ("timestamp", "week_start"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = {
        c: ("sum" if "precipitation" in c or "rain" in c else "mean")
        for c in df.columns if c not in ("timestamp", "week_start")
    }
    weekly = df.groupby("week_start").agg(agg).reset_index()
    rename = {}
    for c in weekly.columns:
        cl = c.lower()
        if "precipitation" in cl:    rename[c] = "precip_sum"
        elif cl == "rain":            rename[c] = "rain_sum"
        elif "temperature" in cl:    rename[c] = "temp_mean"
        elif "relative_humidity" in cl: rename[c] = "humidity_mean"
        elif "cloud_cover" in cl:    rename[c] = "cloud_cover_mean"
    weekly = weekly.rename(columns=rename)
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    return weekly


def fetch_gap_weather(depot: dict, cfg: dict) -> pd.DataFrame:
    """
    Fetch Tier 2 + Tier 3 weather for a depot covering 2022-12-01 to today.
    Returns weekly DataFrame with weather columns.
    """
    lat, lon = depot["lat"], depot["lon"]
    tz = cfg["weather"]["timezone"]
    lag_days = int(cfg["weather"]["tier2_lag_days"])
    today = date.today()
    tier2_end = today - timedelta(days=lag_days)

    base = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": tz,
        "format": "csv",
    }

    frames = []

    # Tier 2: historical forecast archive  2022-12-01 → (today - 92 days)
    if tier2_end > date(2022, 12, 1):
        p2 = {**base,
              "start_date": "2022-12-01",
              "end_date": tier2_end.isoformat()}
        try:
            df2 = _fetch_open_meteo(cfg["weather"]["urls"]["tier2"], p2)
            frames.append(_agg_to_weekly(df2))
            logger.info("[WEATHER] Tier2 OK for %s (%s → %s)",
                        depot["name"], "2022-12-01", tier2_end)
        except Exception as e:
            logger.warning("[WEATHER] Tier2 failed for %s: %s", depot["name"], e)

    # Tier 3: current rolling window  (today - 92 days) → today
    p3 = {**base, "past_days": lag_days, "forecast_days": 0}
    try:
        df3 = _fetch_open_meteo(cfg["weather"]["urls"]["current"], p3)
        frames.append(_agg_to_weekly(df3))
        logger.info("[WEATHER] Tier3 OK for %s", depot["name"])
    except Exception as e:
        logger.warning("[WEATHER] Tier3 failed for %s: %s", depot["name"], e)

    if not frames:
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["week_start"])
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    # Keep only the gap range
    combined = combined[combined["week_start"] >= pd.Timestamp("2022-12-05")]
    return combined


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.db.db import get_client
    sb = get_client()

    # depot name → depot_id map
    depot_rows = sb.table("tc_depots").select("depot_id,name").execute().data
    depot_id_map = {r["name"]: r["depot_id"] for r in depot_rows}

    weather_cols = ["precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean"]

    for depot in cfg["depots"]:
        name = depot["name"]
        depot_id = depot_id_map.get(name)
        if not depot_id:
            logger.warning("[BACKFILL] Unknown depot: %s", name)
            continue

        logger.info("[BACKFILL] === %s ===", name)
        try:
            df = fetch_gap_weather(depot, cfg)
            if df.empty:
                logger.warning("[BACKFILL] No data for %s", name)
                continue

            # Build update rows — only weather columns + PK
            records = []
            for _, row in df.iterrows():
                rec = {"week_start": row["week_start"].strftime("%Y-%m-%d")}
                for col in weather_cols:
                    val = row.get(col)
                    rec[col] = float(val) if pd.notna(val) else None
                records.append(rec)

            # Update in batches of 100
            batch_size = 100
            updated = 0
            for i in range(0, len(records), batch_size):
                batch = records[i: i + batch_size]
                for rec in batch:
                    sb.table("tc_demand_panel").update({
                        k: v for k, v in rec.items() if k != "week_start"
                    }).eq("depot_id", depot_id).eq(
                        "week_start", rec["week_start"]
                    ).execute()
                    updated += 1

            logger.info("[BACKFILL] %s — updated %d rows", name, updated)

        except Exception as e:
            logger.error("[BACKFILL] Failed for %s: %s", name, e)

        time.sleep(3)  # respect Open-Meteo rate limits

    logger.info("[BACKFILL] Done.")


if __name__ == "__main__":
    main()
