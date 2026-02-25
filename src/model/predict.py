"""Inference on new matchups."""

import numpy as np
import pandas as pd

from src.model.dataset import DIFF_COLS, FEATURE_COLS


def predict_matchup(
    model,
    team_1_features: dict | pd.Series,
    team_2_features: dict | pd.Series,
) -> float:
    """Predict win probability for team_1 over team_2.

    Returns probability that team_1 wins (0.0 to 1.0).
    """
    diffs = {}
    diffs["efficiency_diff"] = (
        team_1_features.get("efficiency_ratio", 0) - team_2_features.get("efficiency_ratio", 0)
    )
    diffs["off_efficiency_diff"] = (
        team_1_features.get("off_efficiency", 0) - team_2_features.get("off_efficiency", 0)
    )
    diffs["def_efficiency_diff"] = (
        team_1_features.get("def_efficiency", 0) - team_2_features.get("def_efficiency", 0)
    )
    diffs["momentum_diff"] = (
        team_1_features.get("momentum", 0) - team_2_features.get("momentum", 0)
    )
    diffs["sos_diff"] = (
        team_1_features.get("sos", 0) - team_2_features.get("sos", 0)
    )
    diffs["upset_diff"] = (
        team_1_features.get("upset_propensity", 0) - team_2_features.get("upset_propensity", 0)
    )
    diffs["seed_diff"] = (
        team_1_features.get("seed", 8) - team_2_features.get("seed", 8)
    )
    diffs["adj_o_diff"] = (
        team_1_features.get("adj_o", 0) - team_2_features.get("adj_o", 0)
    )
    diffs["adj_d_diff"] = (
        team_1_features.get("adj_d", 0) - team_2_features.get("adj_d", 0)
    )
    diffs["adj_tempo_diff"] = (
        team_1_features.get("adj_tempo", 0) - team_2_features.get("adj_tempo", 0)
    )
    diffs["luck_diff"] = (
        team_1_features.get("luck", 0) - team_2_features.get("luck", 0)
    )

    X = np.array([[diffs[c] for c in DIFF_COLS]])
    prob = model.predict_proba(X)[0, 1]
    return float(prob)


def predict_all_matchups(
    model,
    features: pd.DataFrame,
    team_ids: list[str],
    seeds: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Predict win probabilities for all possible matchups between teams.

    Returns DataFrame with columns: team_1_id, team_2_id, team_1_win_prob
    """
    if seeds is None:
        seeds = {}

    results = []
    for i, t1 in enumerate(team_ids):
        for t2 in team_ids[i + 1:]:
            t1_feats = features[features["team_id"] == str(t1)]
            t2_feats = features[features["team_id"] == str(t2)]

            if t1_feats.empty or t2_feats.empty:
                continue

            t1_dict = t1_feats.iloc[-1].to_dict()
            t2_dict = t2_feats.iloc[-1].to_dict()
            t1_dict["seed"] = seeds.get(str(t1), 8)
            t2_dict["seed"] = seeds.get(str(t2), 8)

            prob = predict_matchup(model, t1_dict, t2_dict)
            results.append({
                "team_1_id": str(t1),
                "team_2_id": str(t2),
                "team_1_win_prob": prob,
            })

    return pd.DataFrame(results)
