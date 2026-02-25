# Pool Optimizer Guide

The pool optimizer transforms the bracket predictor from a "pick the most likely winner" system into a "maximize your chance of winning a bracket pool" system. In a pool, you don't need to be right — you need to be right *where others are wrong*.

## The Core Insight

In a 285-person pool with standard 10/20/40/80/160/320 scoring, picking the chalk champion gives you roughly a 1/285 chance of winning even if you're right, because everyone else picked them too. Picking a less popular but still viable champion gives you a much higher payout when correct.

The optimizer quantifies this tradeoff using **leverage scoring**.

## Algorithm

### Step 1: Win Probability

For each matchup, the base `predict_matchup()` model produces a win probability. Two adjustments are applied:

**Luck Fade**: Teams with a Barttorvik luck rating above the threshold (default: 3.0) have their win probability multiplied by 0.85. These teams have overperformed their underlying metrics and are likely to regress.

```python
if abs(team.luck) > luck_fade_threshold:
    win_prob *= 0.85
```

After fading, probabilities are renormalized to sum to 1.

### Step 2: Leverage Score

Leverage measures how much the model disagrees with the public:

```
leverage = win_prob / (public_pick_pct / 100)
```

| Leverage | Meaning |
|---|---|
| > 2.0 | Model much higher on team than public — high value |
| 1.0 | Model and public agree |
| < 0.5 | Model much lower than public — potential trap |

### Step 3: Pool-Relative EV

Standard EV: `reach_prob * win_prob * points`

Pool-relative EV divides by expected sharers:

```
expected_sharers = pool_size * (public_pick_pct / 100)
pool_ev = reach_prob * win_prob * points / max(expected_sharers, 1)
```

This means a correct pick shared by 200 people in a 285-person pool is worth very little, while a correct pick shared by 3 people is enormously valuable.

### Step 4: Blended EV

Raw EV and pool EV are blended with round-dependent weights:

```python
ROUND_WEIGHTS = {1: 0.1, 2: 0.2, 3: 0.5, 4: 0.7, 5: 0.9, 6: 1.0}
blended = (1 - w) * raw_ev + w * pool_ev
```

**Rationale**: In early rounds (10 points each), the points at stake are small, so just pick the chalk — getting cute with 12-over-5 upsets costs more in bust risk than it gains in differentiation. In late rounds (160-320 points), differentiation is everything, so pool EV dominates.

### Step 5: Special Rules

**12-vs-5 Exception** (Round 1 only):
If a 12-seed has `win_prob > 0.45` AND `leverage > 1.5`, pick the upset. The 12-5 matchup is historically the most common upset and often underpriced by the public.

**Champion Filter** (Round 6 only):
If `champion_filter` is enabled, only teams flagged as `champion_viable` (top 20 AdjO + AdjD) can be picked as champion. Among viable teams, the one with the highest blended EV wins.

## Configuration

All settings are in `config.yaml` under the `pool` section:

```yaml
pool:
  pool_size: 285           # Number of entrants
  strategy: "leverage"     # "leverage", "pure_ev", or "hybrid"
  champion_filter: true    # Only viable teams can win title
  luck_fade_threshold: 3.0 # Luck rating above which to fade
  early_round_chalk_threshold: 0.55  # Reserved for future use
  upset_leverage_threshold: 2.0      # Reserved for future use
```

These can also be adjusted via the Streamlit sidebar.

### Tuning for Pool Size

| Pool Size | Strategy Notes |
|---|---|
| 10-25 | Small pool. Reduce contrarian picks — chalk usually wins small pools. Consider `strategy: "pure_ev"`. |
| 25-100 | Medium pool. Moderate leverage. The default settings work but consider lowering `pool_size` in config. |
| 100-500 | Large pool. Full leverage optimization. Default settings are calibrated for this range. |
| 500+ | Very large pool. You need maximum differentiation. Consider disabling `champion_filter` if there's a strong non-traditional champion candidate. |

## Public Pick Data

The optimizer needs public pick percentages (what % of the pool picks each team in each round). Three sources are supported, in priority order:

### 1. CSV Upload
Upload a CSV with columns: `team_name, round_1, round_2, ..., round_6`. Values are percentages (0-100).

Sources for this data:
- ESPN Tournament Challenge aggregate picks (available after bracket selection opens)
- CBS/Yahoo pool data
- Action Network or other sports analytics sites

### 2. Seed-Based Defaults
Historical averages based on seed line. Used when no CSV is available. These are baked into `src/data/public_picks.py`:

| Seed | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|
| 1 | 99% | 90% | 65% | 45% | 30% | 25% |
| 2 | 94% | 70% | 40% | 22% | 12% | 8% |
| 3 | 85% | 50% | 25% | 12% | 5% | 3% |
| 4 | 79% | 40% | 18% | 8% | 3% | 2% |
| 5 | 62% | 28% | 12% | 5% | 2% | 1% |
| 8 | 50% | 15% | 5% | 2% | 0.5% | 0.3% |
| 12 | 38% | 10% | 3% | 1% | 0.3% | 0.1% |
| 16 | 1% | 0.2% | 0.05% | 0.01% | — | — |

### 3. Custom per-team
Pass `public_pick_pct` directly in the `bracket_teams` list when using `build_leverage_bracket_from_features()`.

## Streamlit UI

The **Pool Optimizer** tab provides:

1. **Data Loading** — Upload Barttorvik CSV or fetch live data; upload public picks CSV or use seed defaults
2. **Champion Viable Teams** — Table of teams meeting the AdjO/AdjD threshold
3. **Leverage Analysis** — All teams ranked by leverage score with efficiency and public pick %
4. **Generate Bracket** — Runs the leverage builder and displays picks with reasons
5. **Tiebreaker** — Tempo-adjusted championship total prediction
6. **Compare Mode** — Side-by-side diff showing where the leverage bracket diverges from pure EV

## Usage Examples

### Python API

```python
from src.config import Config
from src.data.storage import load_parquet
from src.model.bracket import build_leverage_bracket_from_features
import joblib

config = Config.load()
features = load_parquet(config.processed_data_path / "features.parquet")
model = joblib.load(config.models_path / "bracket_model.joblib")

bracket_teams = [
    {"team_id": "2250", "name": "UConn", "seed": 1},
    {"team_id": "150", "name": "Fairleigh Dickinson", "seed": 16},
    # ... 62 more teams in bracket order
]

picks = build_leverage_bracket_from_features(
    model, features, bracket_teams,
    pool_size=285,
    champion_filter=True,
    luck_fade_threshold=3.0,
)

for pick in picks:
    print(f"R{pick['round']}: {pick['winner']} ({pick['pick_reason']})")
```

### Interpreting Pick Reasons

| Reason | Meaning |
|---|---|
| `chalk` | Favorite picked in rounds 1-2, no leverage needed |
| `model favored` | Standard pick based on blended EV |
| `value upset (leverage 2.3x)` | Model-favored underdog with high leverage |
| `contrarian (leverage 3.1x)` | Late-round pick driven by low public ownership |
| `high leverage (2.5x)` | Mid-round pick where model significantly disagrees with public |

## Tiebreaker

The tiebreaker module predicts the championship game total points:

```
base = 144 (historical average)
tempo_adj = (avg_finalists_tempo - 67.5) / 67.5
predicted_total = 144 * (1 + tempo_adj)
```

Two fast-tempo teams (avg 73) produce ~156 points. Two slow teams (avg 62) produce ~132 points. This is displayed after bracket generation in both the UI and the picks output.
