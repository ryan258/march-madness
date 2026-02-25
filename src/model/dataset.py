"""Build symmetry-augmented matchup pairs for training."""

import pandas as pd


FEATURE_COLS = [
    "efficiency_ratio", "off_efficiency", "def_efficiency",
    "momentum", "sos", "upset_propensity",
    "adj_o", "adj_d", "adj_tempo", "luck",
]

DIFF_COLS = [
    "efficiency_diff", "off_efficiency_diff", "def_efficiency_diff",
    "momentum_diff", "sos_diff", "upset_diff", "seed_diff",
    "adj_o_diff", "adj_d_diff", "adj_tempo_diff", "luck_diff",
]


def build_matchup_dataset(
    tournament_results: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build training dataset from tournament results and features.

    Joins tournament results with features, computes feature deltas,
    and applies symmetry augmentation (each game appears twice with
    teams swapped to remove positional bias).

    Returns DataFrame with diff features and target column 'team_1_win'.
    """
    if tournament_results.empty or features.empty:
        return pd.DataFrame()

    # Ensure consistent types for merging
    features = features.copy()
    features["team_id"] = features["team_id"].astype(str)

    # Use only feature columns that exist in the data
    available_feat_cols = [c for c in FEATURE_COLS if c in features.columns]

    results = tournament_results.copy()
    results["team_1_id"] = results["team_1_id"].astype(str)
    results["team_2_id"] = results["team_2_id"].astype(str)

    # Merge features for team 1
    t1_features = features.rename(
        columns={col: f"{col}_1" for col in available_feat_cols}
    )
    merged = results.merge(
        t1_features[["team_id", "season"] + [f"{c}_1" for c in available_feat_cols]],
        left_on=["team_1_id", "season"],
        right_on=["team_id", "season"],
        how="inner",
    ).drop(columns=["team_id"])

    # Merge features for team 2
    t2_features = features.rename(
        columns={col: f"{col}_2" for col in available_feat_cols}
    )
    merged = merged.merge(
        t2_features[["team_id", "season"] + [f"{c}_2" for c in available_feat_cols]],
        left_on=["team_2_id", "season"],
        right_on=["team_id", "season"],
        how="inner",
    ).drop(columns=["team_id"])

    if merged.empty:
        return pd.DataFrame()

    # Compute feature deltas (team_1 - team_2)
    merged["efficiency_diff"] = merged.get("efficiency_ratio_1", 0) - merged.get("efficiency_ratio_2", 0)
    merged["off_efficiency_diff"] = merged.get("off_efficiency_1", 0) - merged.get("off_efficiency_2", 0)
    merged["def_efficiency_diff"] = merged.get("def_efficiency_1", 0) - merged.get("def_efficiency_2", 0)
    merged["momentum_diff"] = merged.get("momentum_1", 0) - merged.get("momentum_2", 0)
    merged["sos_diff"] = merged.get("sos_1", 0) - merged.get("sos_2", 0)
    merged["upset_diff"] = merged.get("upset_propensity_1", 0) - merged.get("upset_propensity_2", 0)
    merged["seed_diff"] = merged["seed_1"] - merged["seed_2"]
    merged["adj_o_diff"] = merged.get("adj_o_1", 0) - merged.get("adj_o_2", 0)
    merged["adj_d_diff"] = merged.get("adj_d_1", 0) - merged.get("adj_d_2", 0)
    merged["adj_tempo_diff"] = merged.get("adj_tempo_1", 0) - merged.get("adj_tempo_2", 0)
    merged["luck_diff"] = merged.get("luck_1", 0) - merged.get("luck_2", 0)

    # Use only diff columns that were actually computed
    available_diff_cols = [c for c in DIFF_COLS if c in merged.columns]

    # Symmetry augmentation: swap teams
    swapped = merged.copy()
    # Swap the diff signs
    for col in available_diff_cols:
        swapped[col] = -swapped[col]
    swapped["team_1_win"] = 1 - swapped["team_1_win"]

    # Combine original and swapped
    dataset = pd.concat([merged, swapped], ignore_index=True)

    return dataset[available_diff_cols + ["team_1_win", "season", "round"]]
