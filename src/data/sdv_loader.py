"""Bulk historical data loader using sportsdataverse with fallback."""

import logging
from pathlib import Path

import pandas as pd

from src.data.storage import data_exists, load_parquet, save_parquet

logger = logging.getLogger(__name__)

# GitHub release base URL for fallback downloads
SDV_GITHUB_BASE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
)


def load_mbb_team_boxscores(
    seasons: list[int], cache_dir: str | Path
) -> pd.DataFrame:
    """Load team boxscore data for given seasons.

    Tries sportsdataverse first, falls back to direct parquet download.
    """
    cache_path = Path(cache_dir) / "team_boxscores.parquet"
    if data_exists(cache_path):
        logger.info("Loading cached team boxscores from %s", cache_path)
        return load_parquet(cache_path)

    frames = []
    for season in seasons:
        df = _load_season_boxscores(season)
        if df is not None and len(df) > 0:
            frames.append(df)
            logger.info("Loaded %d rows for season %d", len(df), season)

    if not frames:
        logger.warning("No team boxscore data loaded")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    save_parquet(result, cache_path)
    return result


def _load_season_boxscores(season: int) -> pd.DataFrame | None:
    """Load a single season's team boxscores."""
    try:
        from sportsdataverse.mbb import mbb_loaders

        df = mbb_loaders.load_mbb_team_boxscore(seasons=[season])
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df
    except Exception:
        logger.warning(
            "sportsdataverse load failed for %d, trying fallback", season
        )

    return _fallback_download(season, "team_boxscore")


def _fallback_download(season: int, data_type: str) -> pd.DataFrame | None:
    """Download parquet directly from sportsdataverse GitHub releases."""
    url = f"{SDV_GITHUB_BASE}/mbb_{data_type}/mbb_{data_type}_{season}.parquet"
    try:
        df = pd.read_parquet(url)
        return df
    except Exception:
        logger.warning("Fallback download failed for %s %d", data_type, season)
        return None


def load_mbb_schedule(seasons: list[int], cache_dir: str | Path) -> pd.DataFrame:
    """Load schedule data for given seasons."""
    cache_path = Path(cache_dir) / "schedules.parquet"
    if data_exists(cache_path):
        return load_parquet(cache_path)

    frames = []
    for season in seasons:
        try:
            from sportsdataverse.mbb import mbb_loaders

            df = mbb_loaders.load_mbb_schedule(seasons=[season])
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                frames.append(df)
        except Exception:
            df = _fallback_download(season, "schedule")
            if df is not None:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    save_parquet(result, cache_path)
    return result


def load_mbb_player_boxscores(
    seasons: list[int], cache_dir: str | Path
) -> pd.DataFrame:
    """Load player boxscore data for given seasons."""
    cache_path = Path(cache_dir) / "player_boxscores.parquet"
    if data_exists(cache_path):
        return load_parquet(cache_path)

    frames = []
    for season in seasons:
        try:
            from sportsdataverse.mbb import mbb_loaders

            df = mbb_loaders.load_mbb_player_boxscore(seasons=[season])
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                frames.append(df)
        except Exception:
            df = _fallback_download(season, "player_boxscore")
            if df is not None:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    save_parquet(result, cache_path)
    return result
