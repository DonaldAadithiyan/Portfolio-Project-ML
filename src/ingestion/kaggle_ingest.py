import logging
import os
import shutil
import glob
import pandas as pd

logger = logging.getLogger(__name__)


def download_kaggle_dataset(cfg: dict) -> str:
    """Download Kaggle dataset and return path to the main CSV."""
    import kagglehub

    out_dir = cfg["paths"]["raw_kaggle"]
    os.makedirs(out_dir, exist_ok=True)

    # Check if already downloaded
    existing = glob.glob(os.path.join(out_dir, "*.csv"))
    if existing:
        logger.info("[KAGGLE] Already downloaded — skipping. Found: %s", existing[0])
        return existing[0]

    dataset = cfg["kaggle"]["dataset"]
    logger.info("[KAGGLE] Downloading dataset: %s", dataset)
    path = kagglehub.dataset_download(dataset)
    logger.info("[KAGGLE] Downloaded to: %s", path)

    # Copy all CSVs to our raw dir
    for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
        dest = os.path.join(out_dir, os.path.basename(f))
        shutil.copy2(f, dest)
        logger.info("[KAGGLE] Copied %s -> %s", f, dest)

    csvs = glob.glob(os.path.join(out_dir, "*.csv"))
    if not csvs:
        raise RuntimeError("[KAGGLE] No CSV files found after download")

    logger.info("[KAGGLE] Dataset ready at: %s", csvs[0])
    return csvs[0]


def load_kaggle_csv(path: str) -> pd.DataFrame:
    """Load and normalise the Kaggle cement CSV."""
    df = pd.read_csv(path)
    # Normalise column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Parse Month column (format: YYYY-MM or Jan-YY or similar)
    df["Month"] = pd.to_datetime(df["Month"], format="%b-%y")
    df = df.sort_values("Month").reset_index(drop=True)
    logger.info("[KAGGLE] Loaded %d rows, date range %s – %s",
                len(df), df["Month"].min().date(), df["Month"].max().date())
    return df
