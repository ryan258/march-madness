"""Adjusted offensive/defensive efficiency ratio."""

import pandas as pd


def compute_possessions(df: pd.DataFrame) -> pd.Series:
    """Estimate possessions: FGA - OREB + TO + 0.475 * FTA."""
    fga = df.get("field_goals_attempted", df.get("fieldGoalsAttempted", 0))
    oreb = df.get("offensive_rebounds", df.get("offensiveRebounds", 0))
    to = df.get("turnovers", 0)
    fta = df.get("free_throws_attempted", df.get("freeThrowsAttempted", 0))

    return fga - oreb + to + 0.475 * fta


def compute_efficiency(team_boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute efficiency ratio per team per season.

    Efficiency Ratio = Adj. Offensive Efficiency / Adj. Defensive Efficiency
    Offensive Efficiency = Points scored / Possessions * 100
    Defensive Efficiency = Points allowed / Possessions * 100

    Returns DataFrame with columns: team_id, season, efficiency_ratio,
        off_efficiency, def_efficiency
    """
    df = team_boxscores.copy()

    # Normalize column names - handle various naming conventions
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        if lower != col:
            col_map[col] = lower
    if col_map:
        df = df.rename(columns=col_map)

    # Try to identify key columns
    team_id_col = _find_col(df, ["team_id", "team_uid", "teamid"])
    season_col = _find_col(df, ["season", "season_type"])
    points_col = _find_col(df, ["team_score", "points", "pts"])
    opp_points_col = _find_col(
        df, ["opponent_team_score", "opp_points", "opponent_points", "opp_pts"]
    )

    if team_id_col is None or season_col is None:
        raise ValueError(
            f"Cannot find required columns. Available: {list(df.columns)[:20]}"
        )

    possessions = compute_possessions(df)
    possessions = possessions.clip(lower=1)  # avoid division by zero

    points = df[points_col] if points_col else pd.Series(0, index=df.index)

    off_eff = (points / possessions) * 100

    if opp_points_col and opp_points_col in df.columns:
        def_eff = (df[opp_points_col] / possessions) * 100
    else:
        # If no opponent score, use league average as placeholder
        def_eff = pd.Series(100.0, index=df.index)

    df["_off_eff"] = off_eff
    df["_def_eff"] = def_eff

    # Aggregate per team per season
    result = (
        df.groupby([team_id_col, season_col])
        .agg(off_efficiency=("_off_eff", "mean"), def_efficiency=("_def_eff", "mean"))
        .reset_index()
    )
    result = result.rename(columns={team_id_col: "team_id", season_col: "season"})

    # Efficiency ratio (higher is better: good offense, good defense)
    result["def_efficiency"] = result["def_efficiency"].clip(lower=1)
    result["efficiency_ratio"] = result["off_efficiency"] / result["def_efficiency"]

    return result[["team_id", "season", "efficiency_ratio", "off_efficiency", "def_efficiency"]]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    # Try case-insensitive match
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None
