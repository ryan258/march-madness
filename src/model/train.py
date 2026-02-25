"""Model training pipeline: LogisticRegression + optional XGBoost."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.dataset import DIFF_COLS

logger = logging.getLogger(__name__)


def train_logistic(
    dataset: pd.DataFrame,
    model_dir: str | Path,
    cv_folds: int = 5,
) -> Pipeline:
    """Train a LogisticRegression pipeline with GridSearchCV.

    Pipeline: StandardScaler -> LogisticRegression
    Grid search over C=[0.01, 0.1, 1.0, 10.0]

    Saves model to model_dir/bracket_model.joblib
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = dataset[DIFF_COLS].values
    y = dataset["team_1_win"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X, y)

    logger.info("Best C: %s, Best accuracy: %.4f", grid.best_params_, grid.best_score_)

    model = grid.best_estimator_
    model_path = model_dir / "bracket_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    return model


def train_xgboost(
    dataset: pd.DataFrame,
    model_dir: str | Path,
) -> object:
    """Train an XGBoost classifier as an alternative model."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not available, skipping")
        return None

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = dataset[DIFF_COLS].values
    y = dataset["team_1_win"].values

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)

    model_path = model_dir / "bracket_model_xgb.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved XGBoost model to %s", model_path)

    return model
