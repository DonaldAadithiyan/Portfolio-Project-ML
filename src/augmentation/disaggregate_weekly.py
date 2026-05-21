import logging
import os
from calendar import monthrange

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_COLS = ["Production", "Sales", "Demand"]


def _weeks_in_month(year: int, month: int) -> list[pd.Timestamp]:
    """Return list of week-start Mondays that start within this month."""
    first = pd.Timestamp(year, month, 1)
    last = pd.Timestamp(year, month, monthrange(year, month)[1])
    # All Mondays in the range
    mondays = pd.date_range(start=first - pd.Timedelta(days=first.weekday()),
                            end=last, freq="W-MON")
    # Keep only those where the Monday falls within the month
    return [m for m in mondays if m.month == month or m >= first]


def _get_weights(n: int, cfg: dict) -> list[float]:
    aug = cfg["augmentation"]
    if n == 4:
        return list(aug["weekly_weights_4"])
    elif n == 5:
        return list(aug["weekly_weights_5"])
    else:
        return [1.0 / n] * n


def disaggregate_weekly(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Expand monthly rows to weekly rows using within-month demand weights.
    Economic columns are forward-filled (constant within month).
    Reads:  kaggle_scaled.csv DataFrame (monthly)
    Writes: data/interim/kaggle_weekly.csv
    """
    out_path = os.path.join(cfg["paths"]["interim"], "kaggle_weekly.csv")

    rows = []
    df = df.dropna(subset=["Month"]).copy()
    for _, row in df.iterrows():
        year = int(row["Month"].year)
        month = int(row["Month"].month)

        # Build week starts for this month
        first_day = pd.Timestamp(year, month, 1)
        last_day = pd.Timestamp(year, month, monthrange(year, month)[1])
        mondays = pd.date_range(
            start=first_day - pd.Timedelta(days=first_day.weekday()),
            end=last_day + pd.Timedelta(days=7),
            freq="W-MON",
        )
        # Keep only weeks that overlap with the month
        week_starts = [m for m in mondays
                       if m <= last_day and (m + pd.Timedelta(days=6)) >= first_day]

        n = len(week_starts)
        if n == 0:
            continue
        weights = _get_weights(n, cfg)
        if len(weights) < n:
            weights = [1.0 / n] * n
        weights = weights[:n]
        # Renormalise in case of trimming
        total = sum(weights)
        weights = [w / total for w in weights]

        for i, ws in enumerate(week_starts):
            new_row = {"week_start": ws}
            for col in VOL_COLS:
                if col in row.index:
                    new_row[col] = row[col] * weights[i]
            # Forward-fill economic columns
            for col in row.index:
                if col not in VOL_COLS and col != "Month":
                    new_row[col] = row[col]
            rows.append(new_row)

    result = pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)

    os.makedirs(cfg["paths"]["interim"], exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info("[AUG] disaggregate_weekly: %d rows -> %s", len(result), out_path)
    return result
