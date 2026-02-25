# Feature Engineering Reference

The model uses 10 team-level features computed per season. At prediction time, features are differenced (`team_1 - team_2`) to produce 11 matchup-level diff features (including `seed_diff`).

## Feature Summary

| Feature | Source | Higher = | Range |
|---|---|---|---|
| `efficiency_ratio` | Boxscores | Better | 0.5 - 1.5 |
| `off_efficiency` | Boxscores | Better | 80 - 130 |
| `def_efficiency` | Boxscores | Worse (lower is better) | 80 - 130 |
| `momentum` | Boxscores | Better | -20 to +20 |
| `sos` | Boxscores | Tougher schedule | -10 to +10 |
| `upset_propensity` | Boxscores | More upset-prone (mid-majors) | 0.0 - 0.6 |
| `adj_o` | Barttorvik | Better offense | 90 - 130 |
| `adj_d` | Barttorvik | Worse (lower is better) | 85 - 115 |
| `adj_tempo` | Barttorvik | Faster pace | 60 - 75 |
| `luck` | Barttorvik | Luckier (overperforming) | -8 to +8 |

## Detailed Descriptions

### Efficiency Ratio
**Module**: `src/features/efficiency.py`
**Formula**: `off_efficiency / def_efficiency`

The core team quality metric. Computed from boxscore data:
- **Offensive Efficiency** = (Points / Possessions) * 100
- **Defensive Efficiency** = (Opponent Points / Possessions) * 100
- **Possessions** = FGA - OREB + TO + 0.475 * FTA

A ratio above 1.0 means the team scores more efficiently than it allows. Elite teams typically have ratios of 1.15-1.35.

### Momentum
**Module**: `src/features/momentum.py`
**Formula**: Exponentially-weighted average of last 10 game point differentials

Captures recent form heading into the tournament. Uses a decay factor of 0.85, meaning the most recent game has ~6x the weight of the 10th-most-recent game.

```
weights = [0.85^9, 0.85^8, ..., 0.85^1, 0.85^0]
momentum = weighted_average(point_differentials[-10:], weights)
```

A team that won its last 10 games by 15+ points will have strong positive momentum. A team that lost 3 of its last 5 will have negative or near-zero momentum.

### Strength of Schedule (SOS)
**Module**: `src/features/strength_of_schedule.py`
**Formula**: `mean(point_differential * opponent_win_pct)`

Measures the quality of opponents faced. A team that beats strong opponents (high win %) by large margins gets a high SOS. Beating weak teams or losing to strong teams yields lower values.

### Upset Propensity
**Module**: `src/features/upset_propensity.py`
**Formula**: `eFG% * (1 - turnover_rate)` (mid-majors only)

Identifies mid-major teams capable of pulling upsets. The formula combines:
- **eFG%** = (FGM + 0.5 * 3PM) / FGA — shooting efficiency including 3-point premium
- **Turnover Rate** = TO / Possessions — ball security

Power conference teams (ACC, Big 12, Big East, Big Ten, Pac-12, SEC, AAC) are zeroed out since this metric specifically targets underdog upset potential.

### Adjusted Offense (AdjO)
**Source**: Barttorvik
**Module**: `src/features/advanced_metrics.py`

Barttorvik's adjusted offensive efficiency, which accounts for opponent quality and game location. Unlike the raw `off_efficiency` from boxscores, AdjO normalizes against the average team. Values typically range from 95 (poor) to 125 (elite).

### Adjusted Defense (AdjD)
**Source**: Barttorvik

Barttorvik's adjusted defensive efficiency. **Lower is better** — a team allowing fewer points per 100 possessions after adjustment has a lower AdjD. Elite defenses are around 88-92, poor defenses around 105-115.

### Adjusted Tempo
**Source**: Barttorvik

Possessions per 40 minutes, adjusted for opponent. The league average is approximately 67.5. This feature is also used by the tiebreaker module to predict championship game total points.

### Luck
**Source**: Barttorvik

The difference between a team's actual win-loss record and the record predicted by their efficiency metrics. A luck value of +4 means the team won 4 more games than their per-possession metrics would predict. Used by the pool optimizer's luck fade: teams with high luck are regressed toward their "true" quality.

## Derived Metrics

### Champion Viable (Boolean)
**Module**: `src/features/advanced_metrics.py`
**Formula**: `(adj_o_rank <= 20) AND (adj_d_rank <= 20)`

Teams ranked in the top 20 nationally in both adjusted offense and adjusted defense. Historically, nearly all champions meet this threshold. Used by the pool optimizer's champion filter to exclude non-viable championship picks.

## Diff Features (Model Input)

At prediction time, the model receives 11 difference features:

| Diff Feature | Computation |
|---|---|
| `efficiency_diff` | t1.efficiency_ratio - t2.efficiency_ratio |
| `off_efficiency_diff` | t1.off_efficiency - t2.off_efficiency |
| `def_efficiency_diff` | t1.def_efficiency - t2.def_efficiency |
| `momentum_diff` | t1.momentum - t2.momentum |
| `sos_diff` | t1.sos - t2.sos |
| `upset_diff` | t1.upset_propensity - t2.upset_propensity |
| `seed_diff` | t1.seed - t2.seed |
| `adj_o_diff` | t1.adj_o - t2.adj_o |
| `adj_d_diff` | t1.adj_d - t2.adj_d |
| `adj_tempo_diff` | t1.adj_tempo - t2.adj_tempo |
| `luck_diff` | t1.luck - t2.luck |

Positive values generally favor team 1. The exception is `def_efficiency_diff` and `adj_d_diff`, where a negative value (lower defensive efficiency) favors team 1.

## Imputation

Missing feature values are imputed in `builder.py`:
1. **Season median** — grouped by season, fill NaN with that season's median
2. **Global median** — if an entire season is missing, fill with the cross-season median
3. **Zero fill** — any remaining NaN becomes 0

The `champion_viable` boolean defaults to `False` when Barttorvik data is unavailable.

## Adding New Features

To add a new feature:
1. Create a module in `src/features/` returning a DataFrame with `team_id`, `season`, and your feature column(s)
2. Import it in `builder.py` and merge on `(team_id, season)`
3. Add the column name(s) to the `numeric_cols` imputation list in `builder.py`
4. Add the feature name to `FEATURE_COLS` in `dataset.py`
5. Add the diff name to `DIFF_COLS` in `dataset.py`
6. Add the diff computation in both `dataset.py` and `predict.py`
7. Retrain the model
