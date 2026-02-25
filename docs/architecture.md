# Architecture

## Overview

The system is organized into four layers: **data ingestion**, **feature engineering**, **modeling**, and **presentation**. Each layer is independently testable and communicates via parquet files and pandas DataFrames.

## Directory Layout

```
src/
├── config.py          # Configuration dataclasses (Config, ESPNConfig, PoolConfig)
├── data/              # Data ingestion layer
├── features/          # Feature computation layer
├── model/             # ML modeling layer
└── app/               # Presentation layer (Streamlit)
```

## Data Flow

```
┌─────────────────────────────────────────────────┐
│                  Data Sources                    │
│  ESPN API  │  sportsdataverse  │  Barttorvik     │
└─────┬──────┴───────┬──────────┴──────┬──────────┘
      │              │                 │
      ▼              ▼                 ▼
┌───────────┐ ┌─────────────┐ ┌────────────────┐
│ espn_     │ │ sdv_        │ │ barttorvik_    │
│ client.py │ │ loader.py   │ │ client.py      │
└─────┬─────┘ └──────┬──────┘ └───────┬────────┘
      │              │                │
      │              ▼                ▼
      │      ┌──────────────┐ ┌─────────────────┐
      │      │ Team         │ │ team_mapping.py  │
      │      │ Boxscores    │ │ (name → ESPN ID) │
      │      └──────┬───────┘ └───────┬─────────┘
      │             │                 │
      ▼             ▼                 ▼
┌───────────┐ ┌──────────────────────────────────┐
│tournament_│ │         Feature Builder           │
│scraper.py │ │   (builder.py orchestrates all)   │
└─────┬─────┘ │                                    │
      │       │  efficiency  momentum  sos  upset  │
      │       │  advanced_metrics (Barttorvik)     │
      │       └──────────────┬─────────────────────┘
      │                      │
      ▼                      ▼
┌──────────────────────────────────────────┐
│            Matchup Dataset               │
│  (dataset.py: diffs + symmetry aug.)     │
└───────────────────┬──────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   ┌─────────────┐    ┌─────────────┐
   │  train.py   │    │ evaluate.py │
   │ (GridSearch │    │  (LOSO-CV)  │
   │  pipeline)  │    └──────┬──────┘
   └──────┬──────┘           │
          │                  ▼
          ▼           ┌────────────┐
   ┌─────────────┐    │calibration │
   │  predict.py │    │  metrics   │
   └──────┬──────┘    └────────────┘
          │
   ┌──────┴───────────────────────┐
   ▼                              ▼
┌──────────────┐    ┌──────────────────────┐
│ BracketBuilder│    │LeverageBracketBuilder│
│ (pure EV)    │    │(pool-optimized)      │
└──────┬───────┘    └──────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
         ┌────────────────┐
         │  Streamlit UI  │
         │  (5 tabs)      │
         └────────────────┘
```

## Module Responsibilities

### Data Layer (`src/data/`)

| Module | Responsibility |
|---|---|
| `espn_client.py` | Rate-limited ESPN API wrapper with retry logic. Fetches scoreboards, teams, rankings, and game summaries. |
| `sdv_loader.py` | Loads bulk historical boxscore data via the sportsdataverse library, with fallback to direct parquet download from GitHub releases. |
| `tournament_scraper.py` | Scrapes NCAA tournament game results from ESPN. Parses seeds, scores, and determines round number from event metadata. |
| `barttorvik_client.py` | Fetches Barttorvik advanced metrics (AdjO, AdjD, Tempo, Luck, Barthag). Supports CSV fallback and parquet caching. |
| `team_mapping.py` | Maps between Barttorvik display names and ESPN numeric team IDs using normalized string matching and manual overrides. |
| `public_picks.py` | Loads public bracket pick percentages from CSV or provides seed-based historical defaults for all 16 seeds across 6 rounds. |
| `storage.py` | Thin parquet read/write helpers with auto-directory creation. |

### Feature Layer (`src/features/`)

| Module | Responsibility |
|---|---|
| `builder.py` | Orchestrates all feature modules. Loads boxscores, runs each computation, merges on `(team_id, season)`, imputes missing values with season median, and caches to `features.parquet`. |
| `efficiency.py` | Offensive/defensive efficiency from boxscore possessions and points. |
| `momentum.py` | Exponentially-weighted point differential over last 10 games. |
| `strength_of_schedule.py` | Point differential weighted by opponent win percentage. |
| `upset_propensity.py` | eFG% * (1 - turnover rate) for mid-major teams; zero for power conferences. |
| `advanced_metrics.py` | Passes through Barttorvik metrics and computes champion viability (top 20 AdjO + AdjD). |

### Model Layer (`src/model/`)

| Module | Responsibility |
|---|---|
| `dataset.py` | Builds training data: joins tournament results with features, computes 11 diff features, applies symmetry augmentation (2x rows). |
| `train.py` | Trains LogisticRegression (with GridSearchCV over C) and optional XGBoost classifier. |
| `predict.py` | Computes win probability for a single matchup from feature diffs. |
| `evaluate.py` | Leave-one-season-out CV with accuracy, log loss, Brier score, AUC. Generates calibration curves. |
| `bracket.py` | `BracketBuilder` (pure EV-optimal) and `LeverageBracketBuilder` (pool-optimized with leverage, luck fade, champion filter). |
| `tiebreaker.py` | Predicts championship game total points from finalists' tempo. |

### Presentation Layer (`src/app/`)

| Module | Responsibility |
|---|---|
| `streamlit_app.py` | Interactive UI with 5 tabs: Bracket View, Team Stats, Matchup Explorer, Model Performance, Pool Optimizer. |

## Key Design Decisions

### Symmetry Augmentation
Every training matchup is doubled by swapping team 1 and team 2 (negating all diffs, flipping the label). This eliminates positional bias and doubles the effective training set size.

### Feature Diffs, Not Raw Values
The model operates on `team_1_feature - team_2_feature` diffs, not raw feature values. This makes the model invariant to absolute scale and naturally expresses relative matchup strength.

### Parquet Caching
All intermediate data (boxscores, features, tournament results) is cached as parquet files. The `force=True` parameter on `build_features()` bypasses the cache when needed.

### Backward Compatibility
New features (adj_o, adj_d, adj_tempo, luck) default to 0 via `.get(..., 0)` in `predict_matchup()`. Models trained without Barttorvik data continue to work; the new diff columns simply contribute nothing.

### Two Bracket Builders
`BracketBuilder` and `LeverageBracketBuilder` share the same team representation (`BracketTeam`) and output format. The leverage builder adds optional fields (`public_pick_pct`, `champion_viable`) that default to neutral values, preserving full backward compatibility.

## Configuration

All configuration lives in `config.yaml` and is loaded into typed dataclasses:

```python
Config
├── espn: ESPNConfig        # API endpoints, rate limits, retries
├── seasons: list[int]      # Which seasons to process (2018-2025)
├── power_conferences       # Conference list for upset propensity
├── round_points            # Scoring per round (10/20/40/80/160/320)
├── pool: PoolConfig        # Pool size, strategy, thresholds
├── raw_data_path           # data/raw
├── processed_data_path     # data/processed
├── tournament_data_path    # data/tournaments
├── models_path             # models/
└── external_data_path      # data/external (Barttorvik cache)
```

## Testing Strategy

Tests use mock models (trained on random data with seed_diff as the only informative feature) and synthetic DataFrames. No network calls or file I/O in tests.

- `test_bracket.py` - Core bracket builder and prediction
- `test_features.py` - Each feature module in isolation
- `test_espn_client.py` - API client initialization and rate limiting
- `test_leverage.py` - Leverage optimizer, pool EV, champion filter, tiebreaker, backward compat
