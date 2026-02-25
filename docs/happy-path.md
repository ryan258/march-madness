# Happy Path Usage Guide

This guide illustrates the standard "happy path" workflow for using the March Madness Predictor from start to finish. Following these steps will take you from a fresh installation to generating your first optimized bracket.

## 1. Setup and Installation

First, clone the repository and set up a Python virtual environment:

```bash
git clone <repo-url>
cd march-madness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configuration

Before running any data ingestion, verify your configuration settings in `config.yaml`. The defaults are generally sensible for a standard prediction run.

Key settings to check:

- `seasons`: Ensure the list of seasons you wish to analyze is correct.
- `pool`: Adjust pool size and optimizer settings if you are generating a bracket for a specific pool.

## 3. Data Ingestion & Feature Building

The system needs to fetch historical data from ESPN, SportsDataVerse, and Barttorvik to engineer the 10 core features for each team. This process is fully automated.

Run the feature builder:

```bash
python -c "
from src.config import Config
from src.features.builder import build_features

config = Config.load()
# This fetches raw data, computes features, and saves them to processed_data/features.parquet
features = build_features(config)
print(f'Successfully calculated features for {len(features)} team-seasons.')
"
```

## 4. Model Training

With features built, the next step is to construct the historical matchup dataset and train the Logistic Regression model using GridSearchCV.

Run the training pipeline:

```bash
python -c "
from src.config import Config
from src.data.storage import load_parquet
from src.model.dataset import build_matchup_dataset
from src.model.train import train_logistic

config = Config.load()
features = load_parquet(config.processed_data_path / 'features.parquet')
tournament = load_parquet(config.tournament_data_path / 'tournament_results.parquet')

# Augments the dataset with symmetrical diffs for unbiased training
dataset = build_matchup_dataset(tournament, features)

# Trains the primary LogisticRegression model and saves it to models/
model = train_logistic(dataset, config.models_path)
print('Model training complete and saved.')
"
```

## 5. View and Export Brackets via UI

The most interactive way to generate and explore your bracket is through the Streamlit UI. The UI will automatically load your trained model and the latest team features.

Launch the Streamlit app:

```bash
streamlit run src/app/streamlit_app.py
```

### Navigating the UI

- **Bracket View Tab**: Visualizes the predicted tournament tree based on pure Expected Value (EV).
- **Matchup Explorer Tab**: Select any two teams to see the model's head-to-head win probability and feature comparisons.
- **Pool Optimizer Tab**: Utilize the `LeverageBracketBuilder` to generate a contrarian bracket tailored to your pool size, integrating public pick percentages and champion viability filters.
- **Export**: Use the export buttons within the UI to download your optimal brackets as a CSV file to easily transfer them to your pool hosting site.

## 6. Next Steps

- **Experiment**: Try tweaking the `config.yaml` pool settings (e.g., `luck_fade_threshold` or `early_round_chalk_threshold`) and observing how the leverage bracket adapts in the Streamlit UI.
- **Cross-Validation**: If you want to evaluate the model's historical accuracy, run the Leave-One-Season-Out (LOSO) cross-validation script located in `src/model/evaluate.py`.
