"""Exponentially weighted last-10-game performance momentum."""

import numpy as np
import pandas as pd


def compute_momentum(
    team_boxscores: pd.DataFrame,
    n_games: int = 10,
    decay: float = 0.85,
) -> pd.DataFrame:
    """Compute exponentially weighted point differential over last N games.

    Most recent game gets highest weight: decay^0, decay^1, ..., decay^(N-1)

    Returns DataFrame with columns: team_id, season, momentum
    """
    df = team_boxscores.copy()

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        if lower != col:
            col_map[col] = lower
    if col_map:
        df = df.rename(columns=col_map)

    team_id_col = _find_col(df, ["team_id", "team_uid", "teamid"])
    season_col = _find_col(df, ["season"])
    date_col = _find_col(df, ["game_date", "date", "game_date_time"])
    points_col = _find_col(df, ["team_score", "points", "pts"])
    opp_points_col = _find_col(
        df, ["opponent_team_score", "opp_points", "opponent_points", "opp_pts"]
    )

    if team_id_col is None or season_col is None:
        raise ValueError(f"Cannot find required columns. Available: {list(df.columns)[:20]}")

    # Compute point differential
    if points_col and opp_points_col and opp_points_col in df.columns:
        df["_point_diff"] = df[points_col] - df[opp_points_col]
    elif points_col:
        df["_point_diff"] = df[points_col] - 70  # assume average opponent score
    else:
        df["_point_diff"] = 0

    # Sort by date if available
    if date_col and date_col in df.columns:
        df = df.sort_values([team_id_col, season_col, date_col])
    else:
        df = df.sort_values([team_id_col, season_col])

    # Compute weighted momentum per team per season
    results = []
    for (team, season), group in df.groupby([team_id_col, season_col]):
        diffs = group["_point_diff"].values[-n_games:]  # last N games
        n = len(diffs)
        if n == 0:
            results.append({"team_id": team, "season": season, "momentum": 0.0})
            continue

        # Weights: most recent game = decay^0 = 1, oldest = decay^(n-1)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weighted_avg = np.average(diffs, weights=weights)
        results.append({"team_id": team, "season": season, "momentum": weighted_avg})

    return pd.DataFrame(results)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None
