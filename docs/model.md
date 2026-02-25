# Model Training and Evaluation

## Overview

The model predicts the probability that team 1 beats team 2 in a tournament matchup, given the difference in their statistical features. Two model types are supported: Logistic Regression (default) and XGBoost (alternative).

## Training Pipeline

### 1. Build Matchup Dataset

```python
from src.model.dataset import build_matchup_dataset

dataset = build_matchup_dataset(tournament_results, features)
```

**Process**:
1. Join tournament results with features on `(team_id, season)` for both teams
2. Compute 11 difference features (team_1 - team_2)
3. Apply symmetry augmentation: duplicate every row with teams swapped and diffs negated

**Symmetry augmentation** doubles the dataset and ensures the model has no bias toward team 1 or team 2 position. A dataset of 500 tournament games becomes 1000 training rows.

### 2. Train Model

```python
from src.model.train import train_logistic, train_xgboost

model = train_logistic(dataset, "models/")
# or
model = train_xgboost(dataset, "models/")
```

#### Logistic Regression (Default)

| Setting | Value |
|---|---|
| Pipeline | StandardScaler → LogisticRegression |
| Regularization search | C ∈ {0.01, 0.1, 1.0, 10.0} |
| Cross-validation | 5-fold, accuracy scoring |
| Max iterations | 1000 |
| Output | `models/bracket_model.joblib` |

The StandardScaler normalizes all diff features to zero mean and unit variance before regression. GridSearchCV selects the best regularization strength.

#### XGBoost (Alternative)

| Setting | Value |
|---|---|
| Trees | 100 |
| Max depth | 4 |
| Learning rate | 0.1 |
| Eval metric | Log loss |
| Output | `models/bracket_model_xgb.joblib` |

XGBoost can capture non-linear interactions (e.g., momentum matters more for low seeds) but is more prone to overfitting on the relatively small tournament dataset.

### 3. Inference

```python
from src.model.predict import predict_matchup

prob = predict_matchup(model, team_1_features, team_2_features)
# Returns float in [0, 1]: probability team 1 wins
```

The function:
1. Computes all 11 diff features from the two feature dicts
2. Arranges them in `DIFF_COLS` order as a numpy array
3. Calls `model.predict_proba()` and returns the class-1 probability

Missing features default to 0 via `.get(feature_name, 0)`, so models trained on fewer features continue to work.

## Feature Importance

The 11 diff features and their typical importance (from logistic regression coefficients):

| Feature | Expected Direction | Importance |
|---|---|---|
| `seed_diff` | Negative (lower seed = better) | High |
| `efficiency_diff` | Positive | High |
| `adj_o_diff` | Positive | Medium-High |
| `adj_d_diff` | Negative (lower = better defense) | Medium-High |
| `momentum_diff` | Positive | Medium |
| `sos_diff` | Positive | Medium |
| `off_efficiency_diff` | Positive | Medium |
| `def_efficiency_diff` | Negative | Medium |
| `luck_diff` | Positive (but faded in pool optimizer) | Low |
| `adj_tempo_diff` | ~0 (tempo alone isn't predictive) | Low |
| `upset_diff` | Context-dependent | Low |

Exact coefficients depend on the training data. View them in the Streamlit "Model Performance" tab.

## Evaluation

### Leave-One-Season-Out Cross-Validation

```python
from src.model.evaluate import leave_one_season_out_cv, plot_calibration

results = leave_one_season_out_cv(dataset)
plot_calibration(results['y_true'], results['y_prob'], 'models/calibration.png')
```

For each season in the dataset:
1. Train on all other seasons
2. Predict on the held-out season
3. Compute accuracy, log loss, Brier score, and AUC

This gives an unbiased estimate of how the model would have performed on a "new" tournament.

### Performance Targets

| Metric | Target | Baseline (Seed Only) |
|---|---|---|
| Accuracy | > 65% | ~64% |
| Log Loss | < 0.65 | ~0.66 |
| Brier Score | < 0.23 | ~0.24 |
| AUC | > 0.68 | ~0.65 |

The seed-only baseline predicts the lower seed wins with a probability derived purely from seed differential. Beating this baseline demonstrates the model adds value beyond seeding.

### Calibration

A well-calibrated model means: when it predicts 70% win probability, the team actually wins ~70% of the time. The calibration plot (`models/calibration.png`) shows:
- **Left panel**: Predicted probability vs. actual win rate (ideally on the diagonal)
- **Right panel**: Histogram of predicted probabilities (should be spread, not clustered at 0.5)

## Full Training Script

```python
from src.config import Config
from src.data.storage import load_parquet
from src.data.tournament_scraper import scrape_tournament_results
from src.features.builder import build_features
from src.model.dataset import build_matchup_dataset
from src.model.evaluate import leave_one_season_out_cv, plot_calibration
from src.model.train import train_logistic

# Load config
config = Config.load()

# Build features (uses cache if available)
features = build_features(config)

# Scrape tournament results (uses cache if available)
tournament = scrape_tournament_results(config)

# Build matchup dataset
dataset = build_matchup_dataset(tournament, features)
print(f"Training on {len(dataset)} matchup rows")

# Evaluate via LOSO-CV
results = leave_one_season_out_cv(dataset)
print(f"Accuracy: {results['aggregate']['accuracy']:.3f}")
print(f"Log Loss: {results['aggregate']['log_loss']:.3f}")
print(f"AUC: {results['aggregate']['auc']:.3f}")

# Plot calibration
plot_calibration(results['y_true'], results['y_prob'],
                 config.models_path / 'calibration.png')

# Train final model on all data
model = train_logistic(dataset, config.models_path)
print(f"Model saved to {config.models_path}")
```

## Retraining After Feature Changes

When new features are added (e.g., Barttorvik metrics):

1. Rebuild features with `force=True`:
   ```python
   features = build_features(config, force=True)
   ```
2. Rebuild the matchup dataset (picks up new `FEATURE_COLS` and `DIFF_COLS` automatically)
3. Retrain the model — `train.py` reads `DIFF_COLS` dynamically, so no code changes are needed
4. Evaluate to confirm the new features improve performance

The model gracefully handles training data where some seasons have Barttorvik metrics and others don't, because missing values are imputed to 0 (the neutral diff value).
