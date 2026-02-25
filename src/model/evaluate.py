"""Model evaluation: leave-one-season-out CV, calibration, metrics."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.dataset import DIFF_COLS

logger = logging.getLogger(__name__)


def leave_one_season_out_cv(dataset: pd.DataFrame) -> dict:
    """Perform leave-one-season-out cross-validation.

    Returns dict with per-season and aggregate metrics:
        accuracy, log_loss, brier_score, auc
    """
    seasons = sorted(dataset["season"].unique())
    results = []

    all_y_true = []
    all_y_prob = []

    for test_season in seasons:
        train = dataset[dataset["season"] != test_season]
        test = dataset[dataset["season"] == test_season]

        if len(train) < 10 or len(test) < 2:
            continue

        X_train = train[DIFF_COLS].values
        y_train = train["team_1_win"].values
        X_test = test[DIFF_COLS].values
        y_test = test["team_1_win"].values

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        results.append({
            "season": test_season,
            "accuracy": acc,
            "log_loss": ll,
            "brier_score": brier,
            "auc": auc,
            "n_games": len(test),
        })

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

        logger.info(
            "Season %d: acc=%.3f, log_loss=%.3f, brier=%.3f, auc=%.3f (n=%d)",
            test_season, acc, ll, brier, auc, len(test),
        )

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    aggregate = {
        "accuracy": accuracy_score(all_y_true, (all_y_prob >= 0.5).astype(int)),
        "log_loss": log_loss(all_y_true, all_y_prob),
        "brier_score": brier_score_loss(all_y_true, all_y_prob),
        "auc": roc_auc_score(all_y_true, all_y_prob) if len(np.unique(all_y_true)) > 1 else float("nan"),
    }

    return {
        "per_season": results,
        "aggregate": aggregate,
        "y_true": all_y_true,
        "y_prob": all_y_prob,
    }


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Plot and save calibration curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve")
    ax1.legend()

    # Histogram of predictions
    ax2.hist(y_prob, bins=20, edgecolor="black")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predictions")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved calibration plot to %s", output_path)
