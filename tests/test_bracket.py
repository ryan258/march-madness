"""Tests for bracket builder and prediction modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bracket import BracketBuilder, BracketTeam
from src.model.dataset import DIFF_COLS, build_matchup_dataset
from src.model.predict import predict_matchup


@pytest.fixture
def mock_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, len(DIFF_COLS))
    # Higher seed_diff (last column) -> higher win probability
    y = (X[:, -1] < 0).astype(int)  # Lower seed number = better

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    return model


@pytest.fixture
def sample_features():
    return pd.DataFrame({
        "team_id": ["1", "2", "3", "4"],
        "season": [2024, 2024, 2024, 2024],
        "efficiency_ratio": [1.2, 1.1, 0.9, 0.95],
        "off_efficiency": [110, 105, 95, 100],
        "def_efficiency": [92, 95, 105, 105],
        "momentum": [5.0, 3.0, -2.0, 1.0],
        "sos": [3.5, 2.0, -1.0, 0.5],
        "upset_propensity": [0.0, 0.0, 0.5, 0.3],
    })


@pytest.fixture
def sample_tournament():
    return pd.DataFrame({
        "season": [2024, 2024],
        "round": [1, 1],
        "seed_1": [1, 2],
        "team_1_id": ["1", "2"],
        "team_1_name": ["Team A", "Team B"],
        "score_1": [80, 75],
        "seed_2": [16, 15],
        "team_2_id": ["3", "4"],
        "team_2_name": ["Team C", "Team D"],
        "score_2": [60, 70],
        "team_1_win": [1, 1],
    })


def test_predict_matchup(mock_model):
    t1 = {"efficiency_ratio": 1.2, "off_efficiency": 110, "def_efficiency": 92,
           "momentum": 5.0, "sos": 3.5, "upset_propensity": 0.0, "seed": 1}
    t2 = {"efficiency_ratio": 0.9, "off_efficiency": 95, "def_efficiency": 105,
           "momentum": -2.0, "sos": -1.0, "upset_propensity": 0.5, "seed": 16}

    prob = predict_matchup(mock_model, t1, t2)
    assert 0.0 <= prob <= 1.0


def test_build_matchup_dataset(sample_tournament, sample_features):
    dataset = build_matchup_dataset(sample_tournament, sample_features)
    assert not dataset.empty
    # Symmetry augmentation: 2 games * 2 = 4 rows
    assert len(dataset) == 4
    assert "team_1_win" in dataset.columns
    for col in DIFF_COLS:
        assert col in dataset.columns


def test_bracket_builder(mock_model):
    """Test bracket with 4 teams."""
    teams = [
        BracketTeam("1", "Team A", 1, {"efficiency_ratio": 1.2, "off_efficiency": 110,
                     "def_efficiency": 92, "momentum": 5.0, "sos": 3.5, "upset_propensity": 0.0}),
        BracketTeam("4", "Team D", 4, {"efficiency_ratio": 0.95, "off_efficiency": 100,
                     "def_efficiency": 105, "momentum": 1.0, "sos": 0.5, "upset_propensity": 0.3}),
        BracketTeam("2", "Team B", 2, {"efficiency_ratio": 1.1, "off_efficiency": 105,
                     "def_efficiency": 95, "momentum": 3.0, "sos": 2.0, "upset_propensity": 0.0}),
        BracketTeam("3", "Team C", 3, {"efficiency_ratio": 0.9, "off_efficiency": 95,
                     "def_efficiency": 105, "momentum": -2.0, "sos": -1.0, "upset_propensity": 0.5}),
    ]

    builder = BracketBuilder(mock_model)
    picks = builder.build_bracket(teams)

    # 4 teams = 2 first-round games + 1 championship = 3 games
    assert len(picks) == 3
    assert picks[0]["round"] == 1
    assert picks[-1]["round"] == 2

    for pick in picks:
        assert 0.0 <= pick["win_prob"] <= 1.0
        assert pick["ev"] >= 0


def test_bracket_builder_invalid_size(mock_model):
    """Test that non-power-of-2 team count raises error."""
    teams = [
        BracketTeam("1", "Team A", 1, {}),
        BracketTeam("2", "Team B", 2, {}),
        BracketTeam("3", "Team C", 3, {}),
    ]
    builder = BracketBuilder(mock_model)
    with pytest.raises(ValueError):
        builder.build_bracket(teams)
