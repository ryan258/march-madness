"""Predict championship game total points for tiebreaker."""

HISTORICAL_AVG_TOTAL = 144
LEAGUE_AVG_TEMPO = 67.5


def predict_championship_total(
    team_1_features: dict,
    team_2_features: dict,
) -> int:
    """Predict total points in the championship game based on tempo.

    Uses historical average (144 points) adjusted by finalists' tempo
    relative to league average.

    Args:
        team_1_features: Features dict for finalist 1 (must include adj_tempo)
        team_2_features: Features dict for finalist 2 (must include adj_tempo)

    Returns:
        Predicted total points (rounded integer)
    """
    t1_tempo = team_1_features.get("adj_tempo", LEAGUE_AVG_TEMPO)
    t2_tempo = team_2_features.get("adj_tempo", LEAGUE_AVG_TEMPO)

    avg_finalists_tempo = (t1_tempo + t2_tempo) / 2.0
    tempo_adjustment = (avg_finalists_tempo - LEAGUE_AVG_TEMPO) / LEAGUE_AVG_TEMPO

    predicted_total = HISTORICAL_AVG_TOTAL * (1 + tempo_adjustment)
    return round(predicted_total)
