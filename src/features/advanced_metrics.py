"""Advanced metrics from Barttorvik data."""

import pandas as pd


def compute_advanced_metrics(barttorvik_df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced metrics from Barttorvik rankings.

    Passes through: adj_o, adj_d, adj_tempo, luck
    Computes: adj_o_rank, adj_d_rank, champion_viable

    Args:
        barttorvik_df: DataFrame with columns team_id, season, adj_o, adj_d, adj_tempo, luck

    Returns:
        DataFrame with columns:
            team_id, season, adj_o, adj_d, adj_tempo, luck, champion_viable
    """
    if barttorvik_df.empty:
        return pd.DataFrame()

    df = barttorvik_df.copy()

    # Ensure required columns exist
    required = {"team_id", "season", "adj_o", "adj_d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Fill optional columns with defaults
    if "adj_tempo" not in df.columns:
        df["adj_tempo"] = 67.5
    if "luck" not in df.columns:
        df["luck"] = 0.0

    # Compute rankings per season
    # adj_o_rank: rank descending (higher offense = rank 1)
    df["adj_o_rank"] = df.groupby("season")["adj_o"].rank(
        ascending=False, method="min"
    ).astype(int)

    # adj_d_rank: rank ascending (lower defense = rank 1, better)
    df["adj_d_rank"] = df.groupby("season")["adj_d"].rank(
        ascending=True, method="min"
    ).astype(int)

    # Champion viable: top 20 in both offense and defense
    df["champion_viable"] = (df["adj_o_rank"] <= 20) & (df["adj_d_rank"] <= 20)

    output_cols = [
        "team_id", "season", "adj_o", "adj_d", "adj_tempo", "luck",
        "champion_viable",
    ]
    return df[output_cols]
