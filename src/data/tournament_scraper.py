"""Historical NCAA tournament bracket results scraper."""

import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.espn_client import ESPNClient
from src.data.storage import data_exists, load_parquet, save_parquet

logger = logging.getLogger(__name__)

# NCAA tournament round mapping (ESPN group IDs / round numbers)
ROUND_NAMES = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet 16",
    4: "Elite Eight",
    5: "Final Four",
    6: "Championship",
}

# March Madness date ranges by season (approximate tournament windows)
TOURNAMENT_DATES = {
    2018: ("20180313", "20180402"),
    2019: ("20190319", "20190408"),
    2021: ("20210319", "20210405"),
    2022: ("20220317", "20220404"),
    2023: ("20230316", "20230403"),
    2024: ("20240319", "20240408"),
    2025: ("20250318", "20250407"),
}


def scrape_tournament_results(
    config: Config, seasons: list[int] | None = None
) -> pd.DataFrame:
    """Scrape historical tournament results from ESPN API.

    Returns DataFrame with columns:
        season, round, seed_1, team_1_id, team_1_name, score_1,
        seed_2, team_2_id, team_2_name, score_2, team_1_win
    """
    cache_path = config.tournament_data_path / "tournament_results.parquet"
    if data_exists(cache_path):
        return load_parquet(cache_path)

    if seasons is None:
        seasons = [s for s in config.seasons if s in TOURNAMENT_DATES]

    client = ESPNClient(config)
    all_games = []

    for season in seasons:
        logger.info("Scraping tournament results for %d", season)
        games = _scrape_season(client, season)
        all_games.extend(games)

    if not all_games:
        logger.warning("No tournament results scraped")
        return pd.DataFrame()

    df = pd.DataFrame(all_games)
    save_parquet(df, cache_path)
    return df


def _scrape_season(client: ESPNClient, season: int) -> list[dict]:
    """Scrape all tournament games for a single season."""
    games = []

    if season not in TOURNAMENT_DATES:
        logger.warning("No tournament dates configured for %d", season)
        return games

    start_date, end_date = TOURNAMENT_DATES[season]

    # Fetch postseason scoreboard
    data = client.get_scoreboard(
        date=f"{start_date}-{end_date}",
        limit=300,
    )
    if data is None:
        # Try fetching by individual dates
        data = client.get_season_scoreboard(season, season_type=3)

    if data is None:
        logger.warning("No scoreboard data for %d", season)
        return games

    events = data.get("events", [])
    for event in events:
        game = _parse_tournament_game(event, season)
        if game is not None:
            games.append(game)

    logger.info("Found %d tournament games for %d", len(games), season)
    return games


def _parse_tournament_game(event: dict, season: int) -> dict | None:
    """Parse a single tournament game event from ESPN API response."""
    try:
        competition = event["competitions"][0]
        competitors = competition["competitors"]

        if len(competitors) != 2:
            return None

        # Determine home/away (ESPN lists home first, away second)
        teams = {}
        for comp in competitors:
            key = "home" if comp.get("homeAway") == "home" else "away"
            seed_str = comp.get("curatedRank", {}).get("current", "0")
            try:
                seed = int(seed_str)
            except (ValueError, TypeError):
                seed = 0
            teams[key] = {
                "id": int(comp["team"]["id"]),
                "name": comp["team"].get("displayName", "Unknown"),
                "seed": seed,
                "score": int(comp.get("score", 0)),
                "winner": comp.get("winner", False),
            }

        # Use first competitor as team_1
        t1 = teams.get("home", teams.get("away"))
        t2 = teams.get("away", teams.get("home"))
        if t1 is None or t2 is None:
            return None

        # Determine round from notes or season type detail
        round_num = _determine_round(event)

        return {
            "season": season,
            "game_id": event.get("id", ""),
            "round": round_num,
            "seed_1": t1["seed"],
            "team_1_id": t1["id"],
            "team_1_name": t1["name"],
            "score_1": t1["score"],
            "seed_2": t2["seed"],
            "team_2_id": t2["id"],
            "team_2_name": t2["name"],
            "score_2": t2["score"],
            "team_1_win": int(t1["winner"]),
        }
    except (KeyError, IndexError, TypeError):
        logger.debug("Failed to parse event: %s", event.get("id", "unknown"))
        return None


def _determine_round(event: dict) -> int:
    """Determine tournament round number from ESPN event data."""
    # Check competition notes for round info
    notes = event.get("competitions", [{}])[0].get("notes", [])
    for note in notes:
        headline = note.get("headline", "").lower()
        if "championship" in headline and "final four" not in headline:
            return 6
        elif "final four" in headline or "semifinal" in headline:
            return 5
        elif "elite" in headline:
            return 4
        elif "sweet" in headline:
            return 3
        elif "second round" in headline:
            return 2
        elif "first round" in headline:
            return 1

    # Fallback: check status detail
    status = event.get("status", {}).get("type", {}).get("detail", "").lower()
    if "championship" in status:
        return 6

    return 0  # Unknown round
