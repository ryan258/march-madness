"""Strength of schedule: average point differential weighted by opponent quality."""

import pandas as pd


def compute_strength_of_schedule(team_boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute SOS as avg point differential weighted by opponent win percentage.

    Returns DataFrame with columns: team_id, season, sos
    """
    df = team_boxscores.copy()

    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        if lower != col:
            col_map[col] = lower
    if col_map:
        df = df.rename(columns=col_map)

    team_id_col = _find_col(df, ["team_id", "team_uid", "teamid"])
    season_col = _find_col(df, ["season"])
    points_col = _find_col(df, ["team_score", "points", "pts"])
    opp_id_col = _find_col(
        df, ["opponent_team_id", "opponent_id", "opp_team_id", "opponentteamid"]
    )
    opp_points_col = _find_col(
        df, ["opponent_team_score", "opp_points", "opponent_points", "opp_pts"]
    )

    if team_id_col is None or season_col is None:
        raise ValueError(f"Cannot find required columns. Available: {list(df.columns)[:20]}")

    # Compute each team's win percentage per season
    if points_col and opp_points_col and opp_points_col in df.columns:
        df["_win"] = (df[points_col] > df[opp_points_col]).astype(int)
        df["_point_diff"] = df[points_col] - df[opp_points_col]
    else:
        df["_win"] = 0
        df["_point_diff"] = 0

    team_win_pct = (
        df.groupby([team_id_col, season_col])["_win"]
        .mean()
        .reset_index()
        .rename(columns={"_win": "_opp_win_pct", team_id_col: "_opp_id"})
    )

    # Join opponent win percentage
    if opp_id_col and opp_id_col in df.columns:
        df = df.merge(
            team_win_pct,
            left_on=[opp_id_col, season_col],
            right_on=["_opp_id", season_col],
            how="left",
        )
        df["_opp_win_pct"] = df["_opp_win_pct"].fillna(0.5)
    else:
        df["_opp_win_pct"] = 0.5

    # Weighted point differential
    df["_weighted_diff"] = df["_point_diff"] * df["_opp_win_pct"]

    result = (
        df.groupby([team_id_col, season_col])
        .agg(sos=("_weighted_diff", "mean"))
        .reset_index()
        .rename(columns={team_id_col: "team_id", season_col: "season"})
    )

    return result[["team_id", "season", "sos"]]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None
