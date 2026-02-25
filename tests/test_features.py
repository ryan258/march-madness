"""Tests for feature engineering modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.efficiency import compute_efficiency, compute_possessions
from src.features.momentum import compute_momentum
from src.features.strength_of_schedule import compute_strength_of_schedule
from src.features.upset_propensity import compute_upset_propensity


@pytest.fixture
def sample_boxscores():
    """Create sample team boxscore data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "team_id": np.repeat(["team_a", "team_b", "team_c", "team_d"], n // 4),
        "season": [2023] * n,
        "game_date": pd.date_range("2023-01-01", periods=n // 4).tolist() * 4,
        "team_score": np.random.randint(55, 100, n),
        "opponent_team_score": np.random.randint(55, 100, n),
        "opponent_team_id": np.random.choice(["team_a", "team_b", "team_c", "team_d"], n),
        "field_goals_made": np.random.randint(18, 35, n),
        "field_goals_attempted": np.random.randint(50, 75, n),
        "three_point_field_goals_made": np.random.randint(3, 15, n),
        "free_throws_attempted": np.random.randint(10, 30, n),
        "offensive_rebounds": np.random.randint(5, 15, n),
        "turnovers": np.random.randint(8, 20, n),
        "team_conference": ["ACC"] * (n // 4) + ["ACC"] * (n // 4) + ["Big 12"] * (n // 4) + ["MVC"] * (n // 4),
    })


def test_compute_possessions(sample_boxscores):
    poss = compute_possessions(sample_boxscores)
    assert len(poss) == len(sample_boxscores)
    assert (poss > 0).all()


def test_compute_efficiency(sample_boxscores):
    result = compute_efficiency(sample_boxscores)
    assert "team_id" in result.columns
    assert "season" in result.columns
    assert "efficiency_ratio" in result.columns
    assert len(result) == 4  # 4 teams
    assert (result["efficiency_ratio"] > 0).all()


def test_compute_momentum(sample_boxscores):
    result = compute_momentum(sample_boxscores)
    assert "team_id" in result.columns
    assert "momentum" in result.columns
    assert len(result) == 4


def test_compute_sos(sample_boxscores):
    result = compute_strength_of_schedule(sample_boxscores)
    assert "team_id" in result.columns
    assert "sos" in result.columns
    assert len(result) == 4


def test_compute_upset_propensity(sample_boxscores):
    result = compute_upset_propensity(sample_boxscores)
    assert "team_id" in result.columns
    assert "upset_propensity" in result.columns
    assert len(result) == 4
    # Power conference teams (ACC, Big 12) should have 0 upset propensity
    acc_teams = result[result["team_id"].isin(["team_a", "team_b"])]
    assert (acc_teams["upset_propensity"] == 0.0).all()
    # MVC team should have non-zero upset propensity
    mvc_team = result[result["team_id"] == "team_d"]
    assert (mvc_team["upset_propensity"] > 0).all()
