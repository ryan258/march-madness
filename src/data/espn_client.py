"""Rate-limited ESPN API wrapper for NCAA Men's Basketball."""

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import Config

logger = logging.getLogger(__name__)


class ESPNClient:
    """ESPN API client with retry logic and rate limiting."""

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        retry = Retry(
            total=config.espn.max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        wait = self.config.espn.rate_limit_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict | None = None) -> dict | None:
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                logger.warning("404 for %s", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            logger.exception("Request failed: %s", url)
            return None

    def get_scoreboard(self, date: str, limit: int = 200) -> dict | None:
        """Fetch scoreboard for a given date (YYYYMMDD)."""
        return self._get(
            self.config.espn.scoreboard_url,
            params={"dates": date, "limit": str(limit)},
        )

    def get_team(self, team_id: int) -> dict | None:
        """Fetch team info by ESPN team ID."""
        return self._get(f"{self.config.espn.teams_url}/{team_id}")

    def get_rankings(self, season: int, week: int | None = None) -> dict | None:
        """Fetch rankings for a season (optionally a specific week)."""
        params = {"season": str(season)}
        if week is not None:
            params["week"] = str(week)
        return self._get(self.config.espn.rankings_url, params=params)

    def get_teams(self, limit: int = 400) -> dict | None:
        """Fetch all teams."""
        return self._get(self.config.espn.teams_url, params={"limit": str(limit)})

    def get_game_summary(self, game_id: str) -> dict | None:
        """Fetch game summary/boxscore."""
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball/"
            f"mens-college-basketball/summary?event={game_id}"
        )
        return self._get(url)

    def get_season_scoreboard(
        self, season: int, season_type: int = 3, limit: int = 200
    ) -> dict | None:
        """Fetch tournament games for a season.

        season_type: 2=regular season, 3=postseason
        """
        return self._get(
            self.config.espn.scoreboard_url,
            params={
                "dates": str(season),
                "seasontype": str(season_type),
                "limit": str(limit),
            },
        )
