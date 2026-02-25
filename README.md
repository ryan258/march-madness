# March Madness Bracket Predictor

A machine learning system that predicts NCAA Men's Basketball tournament outcomes and builds EV-optimal brackets. Includes a game-theory pool optimizer that maximizes your edge over the field in bracket pools by incorporating public pick percentages, contrarian leverage, and champion viability filters.

## Quick Start

```bash
# Clone and set up
git clone <repo-url> && cd march-madness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build features from historical data
python -c "
from src.config import Config
from src.features.builder import build_features
config = Config.load()
build_features(config)
"

# Train the model
python -c "
from src.config import Config
from src.data.storage import load_parquet
from src.model.dataset import build_matchup_dataset
from src.model.train import train_logistic
config = Config.load()
features = load_parquet(config.processed_data_path / 'features.parquet')
tournament = load_parquet(config.tournament_data_path / 'tournament_results.parquet')
dataset = build_matchup_dataset(tournament, features)
train_logistic(dataset, config.models_path)
"

# Launch the UI
streamlit run src/app/streamlit_app.py
```

## Features

- **10 statistical features** per team: efficiency ratio, offensive/defensive efficiency, momentum, strength of schedule, upset propensity, adjusted offense, adjusted defense, adjusted tempo, and luck
- **EV-optimal bracket builder** that maximizes expected tournament points using recursive simulation
- **Pool optimizer** with game-theory leverage scoring for bracket pools of any size
- **Tiebreaker predictor** using tempo-adjusted championship total
- **Streamlit UI** with bracket visualization, team stats, matchup explorer, model performance, and pool optimizer tabs
- **Leave-one-season-out cross-validation** for unbiased model evaluation

## Project Structure

```
march-madness/
├── config.yaml                  # All configuration (ESPN, scoring, pool settings)
├── requirements.txt
├── src/
│   ├── config.py                # Config + PoolConfig dataclasses
│   ├── data/
│   │   ├── espn_client.py       # ESPN API wrapper (rate-limited, retries)
│   │   ├── sdv_loader.py        # sportsdataverse bulk data loader
│   │   ├── tournament_scraper.py # Historical tournament results
│   │   ├── barttorvik_client.py # Barttorvik advanced metrics client
│   │   ├── team_mapping.py      # Barttorvik <-> ESPN name mapping
│   │   ├── public_picks.py      # Public pick percentages loader
│   │   └── storage.py           # Parquet I/O helpers
│   ├── features/
│   │   ├── builder.py           # Feature pipeline orchestrator
│   │   ├── efficiency.py        # Offensive/defensive efficiency
│   │   ├── momentum.py          # Exponentially-weighted recent form
│   │   ├── strength_of_schedule.py
│   │   ├── upset_propensity.py  # Mid-major upset potential
│   │   └── advanced_metrics.py  # Barttorvik AdjO/AdjD/Tempo/Luck
│   ├── model/
│   │   ├── dataset.py           # Matchup dataset with symmetry augmentation
│   │   ├── train.py             # LogisticRegression + XGBoost training
│   │   ├── predict.py           # Matchup inference
│   │   ├── evaluate.py          # LOSO-CV, calibration curves
│   │   ├── bracket.py           # BracketBuilder + LeverageBracketBuilder
│   │   └── tiebreaker.py        # Championship total prediction
│   └── app/
│       └── streamlit_app.py     # Interactive UI (5 tabs)
├── tests/
│   ├── test_bracket.py
│   ├── test_features.py
│   ├── test_espn_client.py
│   └── test_leverage.py
└── docs/
    ├── architecture.md          # System architecture and data flow
    ├── features.md              # Feature engineering reference
    ├── pool-optimizer.md        # Game-theory pool strategy guide
    ├── data-sources.md          # Data sources and ingestion
    └── model.md                 # Model training and evaluation
```

## Data Pipeline

```
ESPN API / sportsdataverse / Barttorvik
              ↓
    Raw team boxscores + advanced metrics
              ↓
    Feature engineering (10 features per team)
              ↓
    Tournament results scraper (historical brackets)
              ↓
    Matchup dataset (symmetry-augmented diffs)
              ↓
    Model training (LogReg / XGBoost + GridSearchCV)
              ↓
    Bracket builder (pure EV or leverage-optimized)
              ↓
    Streamlit visualization + CSV export
```

## Pool Optimizer

For bracket pools, the system goes beyond raw win probability. The `LeverageBracketBuilder` maximizes your *relative* edge over other pool entrants:

- **Leverage scoring**: `win_prob / public_pick_pct` identifies undervalued teams
- **Pool-relative EV**: accounts for how many entrants share your pick
- **Round-weighted blending**: chalk in early rounds, contrarian in late rounds
- **Champion filter**: only championship-viable teams (top 20 AdjO + AdjD) can win the title
- **Luck fade**: teams with unsustainably high luck ratings get penalized
- **12-vs-5 exception**: model-favored 12-seeds with high leverage get the upset pick

Configure pool settings in `config.yaml` or via the Streamlit sidebar.

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

See the `docs/` directory for detailed guides:

- [Architecture](docs/architecture.md) - System design, data flow, module responsibilities
- [Features](docs/features.md) - All 10 features with formulas and rationale
- [Pool Optimizer](docs/pool-optimizer.md) - Game-theory strategy, leverage algorithm, configuration
- [Data Sources](docs/data-sources.md) - ESPN, sportsdataverse, Barttorvik ingestion
- [Model](docs/model.md) - Training pipeline, evaluation, feature importance
