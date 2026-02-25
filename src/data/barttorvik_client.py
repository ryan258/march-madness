"""Rate-limited Barttorvik API client for advanced team metrics."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.storage import save_parquet

logger = logging.getLogger(__name__)


class BarttovikClient:
    """Barttorvik API client with retry logic and rate limiting."""

    RATE_LIMIT_SECONDS = 1.0
    MAX_RETRIES = 3

    def __init__(self):
        self.session = requests.Session()
        retry = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        wait = self.RATE_LIMIT_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def fetch_rankings(self, season: int) -> pd.DataFrame:
        """Fetch team rankings from Barttorvik for a given season.

        Returns DataFrame with columns:
            team_name, adj_o, adj_d, adj_tempo, barthag, luck, conf
        """
        self._rate_limit()
        url = "https://barttorvik.com/trank.php"
        params = {"year": season, "json": 1}
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                logger.warning("404 for Barttorvik season %d", season)
                return pd.DataFrame()
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            logger.exception("Barttorvik request failed for season %d", season)
            return pd.DataFrame()

        if not data:
            logger.warning("Empty response from Barttorvik for season %d", season)
            return pd.DataFrame()

        rows = []
        for entry in data:
            try:
                rows.append({
                    "team_name": entry[0] if isinstance(entry, list) else entry.get("team", ""),
                    "adj_o": float(entry[5] if isinstance(entry, list) else entry.get("adjoe", 0)),
                    "adj_d": float(entry[6] if isinstance(entry, list) else entry.get("adjde", 0)),
                    "adj_tempo": float(entry[19] if isinstance(entry, list) else entry.get("adj_tempo", 0)),
                    "barthag": float(entry[3] if isinstance(entry, list) else entry.get("barthag", 0)),
                    "luck": float(entry[41] if isinstance(entry, list) else entry.get("luck", 0)),
                    "conf": entry[2] if isinstance(entry, list) else entry.get("conf", ""),
                })
            except (IndexError, KeyError, ValueError, TypeError):
                continue

        df = pd.DataFrame(rows)
        df["season"] = season
        logger.info("Fetched %d teams from Barttorvik for season %d", len(df), season)
        return df

    def fetch_and_cache(self, season: int, cache_dir: Path) -> pd.DataFrame:
        """Fetch rankings and cache to parquet."""
        cache_path = cache_dir / f"barttorvik_{season}.parquet"
        df = self.fetch_rankings(season)
        if not df.empty:
            save_parquet(df, cache_path)
            logger.info("Cached Barttorvik data to %s", cache_path)
        return df

    @staticmethod
    def load_from_csv(path: Path) -> pd.DataFrame:
        """Load Barttorvik data from CSV as fallback.

        Validates required columns: adj_o, adj_d, luck, adj_tempo
        """
        path = Path(path)
        if not path.exists():
            logger.warning("CSV file not found: %s", path)
            return pd.DataFrame()

        df = pd.read_csv(path)
        required = {"adj_o", "adj_d", "luck", "adj_tempo"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        return df
