"""Parquet read/write helpers."""

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def data_exists(path: str | Path) -> bool:
    return Path(path).exists()
