"""Load public pick percentages for bracket pool optimization."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Historical average public pick percentages by seed and round
# Based on ESPN Tournament Challenge data trends
SEED_DEFAULTS: dict[int, dict[int, float]] = {
    1:  {1: 99.0, 2: 90.0, 3: 65.0, 4: 45.0, 5: 30.0, 6: 25.0},
    2:  {1: 94.0, 2: 70.0, 3: 40.0, 4: 22.0, 5: 12.0, 6: 8.0},
    3:  {1: 85.0, 2: 50.0, 3: 25.0, 4: 12.0, 5: 5.0,  6: 3.0},
    4:  {1: 79.0, 2: 40.0, 3: 18.0, 4: 8.0,  5: 3.0,  6: 2.0},
    5:  {1: 62.0, 2: 28.0, 3: 12.0, 4: 5.0,  5: 2.0,  6: 1.0},
    6:  {1: 63.0, 2: 30.0, 3: 13.0, 4: 5.0,  5: 2.0,  6: 1.0},
    7:  {1: 60.0, 2: 22.0, 3: 8.0,  4: 3.0,  5: 1.0,  6: 0.5},
    8:  {1: 50.0, 2: 15.0, 3: 5.0,  4: 2.0,  5: 0.5,  6: 0.3},
    9:  {1: 50.0, 2: 12.0, 3: 4.0,  4: 1.5,  5: 0.4,  6: 0.2},
    10: {1: 40.0, 2: 15.0, 3: 5.0,  4: 2.0,  5: 0.5,  6: 0.3},
    11: {1: 37.0, 2: 12.0, 3: 5.0,  4: 2.0,  5: 0.5,  6: 0.3},
    12: {1: 38.0, 2: 10.0, 3: 3.0,  4: 1.0,  5: 0.3,  6: 0.1},
    13: {1: 21.0, 2: 5.0,  3: 1.5,  4: 0.5,  5: 0.1,  6: 0.05},
    14: {1: 15.0, 2: 3.0,  3: 0.8,  4: 0.2,  5: 0.05, 6: 0.02},
    15: {1: 6.0,  2: 1.0,  3: 0.3,  4: 0.1,  5: 0.02, 6: 0.01},
    16: {1: 1.0,  2: 0.2,  3: 0.05, 4: 0.01, 5: 0.005, 6: 0.002},
}


class PublicPicksLoader:
    """Load and manage public pick percentages."""

    @staticmethod
    def load_from_csv(path: str | Path) -> pd.DataFrame:
        """Load public pick percentages from CSV.

        Expected columns: team_name, round_1, round_2, ..., round_6
        Values are percentages (0-100).
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Public picks CSV not found: %s", path)
            return pd.DataFrame()

        df = pd.read_csv(path)
        required = {"team_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Normalize column names
        round_cols = {}
        for col in df.columns:
            if col.startswith("round_"):
                try:
                    r = int(col.split("_")[1])
                    round_cols[col] = r
                except (ValueError, IndexError):
                    continue

        logger.info(
            "Loaded public picks for %d teams, %d rounds",
            len(df), len(round_cols),
        )
        return df

    @staticmethod
    def create_seed_based_defaults() -> dict[int, dict[int, float]]:
        """Return historical average public pick percentages by seed.

        Returns:
            dict mapping seed -> {round -> pick_percentage}
        """
        return SEED_DEFAULTS.copy()

    @staticmethod
    def get_pick_pct_for_team(
        seed: int,
        team_name: str | None = None,
        csv_data: pd.DataFrame | None = None,
    ) -> dict[int, float]:
        """Get per-round public pick percentages for a team.

        Tries CSV data first, falls back to seed-based defaults.
        """
        # Try CSV lookup
        if csv_data is not None and not csv_data.empty and team_name:
            match = csv_data[
                csv_data["team_name"].str.lower() == team_name.lower()
            ]
            if not match.empty:
                row = match.iloc[0]
                pcts = {}
                for r in range(1, 7):
                    col = f"round_{r}"
                    if col in row.index:
                        pcts[r] = float(row[col])
                if pcts:
                    return pcts

        # Fall back to seed defaults
        return SEED_DEFAULTS.get(seed, SEED_DEFAULTS[8])
