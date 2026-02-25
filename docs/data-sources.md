# Data Sources

The system ingests data from three external sources, each handled by a dedicated client module.

## 1. ESPN API

**Module**: `src/data/espn_client.py`
**Used by**: `src/data/tournament_scraper.py`

The ESPN public API provides tournament game results, team info, and rankings.

### Endpoints

| Endpoint | Method | Usage |
|---|---|---|
| Scoreboard | `get_scoreboard(date)` | Fetch games on a specific date |
| Season Scoreboard | `get_season_scoreboard(season)` | Fetch all postseason games for a year |
| Teams | `get_teams()` | Fetch all team IDs and names |
| Team | `get_team(team_id)` | Fetch a single team's info |
| Rankings | `get_rankings(season)` | Fetch weekly rankings |
| Game Summary | `get_game_summary(game_id)` | Fetch boxscore for a single game |

### Rate Limiting

- 0.5 seconds between requests (configurable via `config.yaml`)
- Automatic retry with exponential backoff on 429/5xx errors
- Maximum 3 retries per request

### Tournament Scraper

`tournament_scraper.py` uses the ESPN client to build a historical dataset of tournament games. It:

1. Fetches postseason scoreboards for each configured season
2. Parses each game event to extract seeds, scores, and team IDs
3. Determines the tournament round from ESPN event metadata (notes/headlines)
4. Caches results to `data/tournaments/tournament_results.parquet`

**Configured seasons**: 2018, 2019, 2021-2025 (2020 cancelled due to COVID)

### Output Schema: `tournament_results.parquet`

| Column | Type | Description |
|---|---|---|
| season | int | Tournament year |
| game_id | str | ESPN game identifier |
| round | int | 1-6 (First Round through Championship) |
| seed_1 | int | Team 1 seed |
| team_1_id | int | ESPN team ID |
| team_1_name | str | Display name |
| score_1 | int | Final score |
| seed_2 | int | Team 2 seed |
| team_2_id | int | ESPN team ID |
| team_2_name | str | Display name |
| score_2 | int | Final score |
| team_1_win | int | 1 if team 1 won, 0 otherwise |

## 2. sportsdataverse

**Module**: `src/data/sdv_loader.py`

The [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-py) library provides bulk historical boxscore data for NCAA Men's Basketball.

### Data Types

| Function | Data | Cache File |
|---|---|---|
| `load_mbb_team_boxscores()` | Per-team per-game stats | `data/raw/team_boxscores.parquet` |
| `load_mbb_schedule()` | Game schedules | `data/raw/schedules.parquet` |
| `load_mbb_player_boxscores()` | Per-player per-game stats | `data/raw/player_boxscores.parquet` |

### Fallback Strategy

1. Try `sportsdataverse.mbb.mbb_loaders` Python API
2. If import fails or returns empty data, download parquet directly from the [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) GitHub releases

### Key Boxscore Columns Used

The feature modules handle flexible column naming (camelCase, snake_case, etc.), but the primary columns used are:

- `team_id` / `team_uid` — Team identifier
- `season` — Season year
- `game_date` — Date for sorting (momentum)
- `team_score` / `points` — Points scored
- `opponent_team_score` — Points allowed
- `opponent_team_id` — For SOS computation
- `field_goals_made`, `field_goals_attempted` — Shooting stats
- `three_point_field_goals_made` — For eFG% in upset propensity
- `free_throws_attempted` — For possession estimation
- `offensive_rebounds` — For possession estimation
- `turnovers` — For possession estimation and upset propensity
- `team_conference` — For power conference classification

## 3. Barttorvik

**Module**: `src/data/barttorvik_client.py`

[Barttorvik](https://barttorvik.com/) provides advanced analytics for college basketball, including adjusted efficiency metrics that account for opponent quality and game location.

### Fetching Data

```python
from src.data.barttorvik_client import BarttovikClient

client = BarttovikClient()
df = client.fetch_rankings(2025)  # Returns DataFrame
df = client.fetch_and_cache(2025, Path("data/external"))  # Fetches + saves parquet
```

### Rate Limiting

- 1.0 second between requests
- Retry with backoff on 429/5xx
- Maximum 3 retries

### Output Schema

| Column | Type | Description |
|---|---|---|
| team_name | str | Display name (Barttorvik format) |
| adj_o | float | Adjusted offensive efficiency |
| adj_d | float | Adjusted defensive efficiency |
| adj_tempo | float | Adjusted tempo (possessions per 40 min) |
| barthag | float | Power rating (0-1 scale) |
| luck | float | Wins above/below expectation |
| conf | str | Conference name |
| season | int | Season year |

### CSV Fallback

If the API is unavailable, you can load data from a CSV:

```python
df = BarttovikClient.load_from_csv("path/to/barttorvik_2025.csv")
```

Required CSV columns: `adj_o`, `adj_d`, `luck`, `adj_tempo`

### Team Name Mapping

Barttorvik uses display names ("UConn") while ESPN uses numeric IDs. The `team_mapping.py` module bridges them:

```python
from src.data.team_mapping import build_name_to_id_mapping, merge_barttorvik_with_ids

# Build mapping from ESPN team data
espn_teams = pd.DataFrame({"team_id": ["41"], "team_name": ["Connecticut"]})
mapping = build_name_to_id_mapping(espn_teams)

# Merge Barttorvik data with ESPN IDs
barttorvik_df = merge_barttorvik_with_ids(barttorvik_df, mapping)
```

Name matching uses:
1. Exact lowercase match
2. Manual overrides (e.g., "Connecticut" ↔ "UConn", "NC State" ↔ "North Carolina State")
3. Normalized match (strip St./State, parenthetical suffixes)

Unmatched teams are logged as warnings.

### Caching

Fetched data is cached to `data/external/barttorvik_{season}.parquet`. The feature builder automatically checks for cached Barttorvik data when building features.

## Data Directory Structure

```
data/
├── raw/                           # sportsdataverse cache
│   ├── team_boxscores.parquet
│   ├── schedules.parquet
│   └── player_boxscores.parquet
├── processed/                     # Feature output
│   └── features.parquet
├── tournaments/                   # Tournament results
│   └── tournament_results.parquet
└── external/                      # Barttorvik cache
    ├── barttorvik_2024.parquet
    └── barttorvik_2025.parquet
```

All data directories are git-ignored. Run the pipeline to regenerate.
