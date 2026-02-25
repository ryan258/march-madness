"""Upset propensity metric for mid-major teams."""

import pandas as pd


POWER_CONFERENCES = {"ACC", "Big 12", "Big East", "Big Ten", "Pac-12", "SEC", "AAC"}


def compute_upset_propensity(
    team_boxscores: pd.DataFrame,
    power_conferences: set[str] | None = None,
) -> pd.DataFrame:
    """Compute upset propensity: eFG% * (1 - turnover_rate) for mid-majors.

    eFG% = (FGM + 0.5 * 3PM) / FGA
    Turnover rate = TO / possessions

    Power conference teams get a score of 0.

    Returns DataFrame with columns: team_id, season, upset_propensity
    """
    if power_conferences is None:
        power_conferences = POWER_CONFERENCES

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
    conf_col = _find_col(
        df, ["team_conference", "conference", "conference_name", "team_conference_name"]
    )
    fgm_col = _find_col(df, ["field_goals_made", "fieldgoalsmade", "fgm"])
    fga_col = _find_col(df, ["field_goals_attempted", "fieldgoalsattempted", "fga"])
    fg3m_col = _find_col(
        df, ["three_point_field_goals_made", "threepointfieldgoalsmade", "fg3m", "three_point_makes"]
    )
    to_col = _find_col(df, ["turnovers", "to"])

    if team_id_col is None or season_col is None:
        raise ValueError(f"Cannot find required columns. Available: {list(df.columns)[:20]}")

    # Compute eFG%
    fgm = df[fgm_col].fillna(0) if fgm_col else pd.Series(0, index=df.index)
    fga = df[fga_col].fillna(1) if fga_col else pd.Series(1, index=df.index)
    fg3m = df[fg3m_col].fillna(0) if fg3m_col else pd.Series(0, index=df.index)
    to = df[to_col].fillna(0) if to_col else pd.Series(0, index=df.index)

    fga_safe = fga.clip(lower=1)
    efg_pct = (fgm + 0.5 * fg3m) / fga_safe

    # Estimate possessions for turnover rate
    fta_col = _find_col(df, ["free_throws_attempted", "freethrowsattempted", "fta"])
    oreb_col = _find_col(df, ["offensive_rebounds", "offensiverebounds", "oreb"])
    fta = df[fta_col].fillna(0) if fta_col else pd.Series(0, index=df.index)
    oreb = df[oreb_col].fillna(0) if oreb_col else pd.Series(0, index=df.index)
    possessions = (fga - oreb + to + 0.475 * fta).clip(lower=1)
    to_rate = to / possessions

    df["_efg_pct"] = efg_pct
    df["_to_rate"] = to_rate
    df["_upset_raw"] = efg_pct * (1 - to_rate)

    # Aggregate per team per season
    group_cols = [team_id_col, season_col]

    result = (
        df.groupby(group_cols)
        .agg(_upset_raw=("_upset_raw", "mean"))
        .reset_index()
        .rename(columns={team_id_col: "team_id", season_col: "season"})
    )

    # Zero out power conference teams
    if conf_col and conf_col in df.columns:
        # Determine each team's conference by mode (most common)
        team_conf = (
            df.groupby([team_id_col, season_col])[conf_col]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
            .reset_index()
            .rename(columns={team_id_col: "team_id", season_col: "season", conf_col: "_conf"})
        )
        result = result.merge(team_conf, on=["team_id", "season"], how="left")
        result.loc[result["_conf"].isin(power_conferences), "_upset_raw"] = 0.0
        result = result.drop(columns=["_conf"])

    result = result.rename(columns={"_upset_raw": "upset_propensity"})
    return result[["team_id", "season", "upset_propensity"]]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None
