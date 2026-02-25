"""Map Barttorvik team names to ESPN team IDs."""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Known mismatches between Barttorvik and ESPN naming
MANUAL_OVERRIDES = {
    "Connecticut": "UConn",
    "UConn": "Connecticut",
    "NC State": "North Carolina State",
    "North Carolina St.": "North Carolina State",
    "LSU": "Louisiana State",
    "SMU": "Southern Methodist",
    "UCF": "Central Florida",
    "UCSB": "UC Santa Barbara",
    "USC": "Southern California",
    "VCU": "Virginia Commonwealth",
    "UNI": "Northern Iowa",
    "UNCW": "UNC Wilmington",
    "ETSU": "East Tennessee State",
    "MTSU": "Middle Tennessee",
    "FDU": "Fairleigh Dickinson",
    "LIU": "Long Island University",
    "UMBC": "Maryland-Baltimore County",
    "SIU Edwardsville": "SIU-Edwardsville",
    "St. John's": "Saint John's",
    "St. Mary's": "Saint Mary's",
    "St. Peter's": "Saint Peter's",
    "St. Bonaventure": "Saint Bonaventure",
    "St. Thomas": "Saint Thomas",
    "Miami FL": "Miami",
    "Miami (FL)": "Miami",
    "Miami (OH)": "Miami (OH)",
}


def _normalize_name(name: str) -> str:
    """Normalize a team name for fuzzy matching."""
    name = name.strip().lower()
    # Remove common suffixes/prefixes
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)
    # Normalize St./Saint
    name = re.sub(r"\bst\.\s*", "saint ", name)
    name = re.sub(r"\bstate\b", "st", name)
    # Remove extra whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_name_to_id_mapping(espn_teams_df: pd.DataFrame) -> dict[str, str]:
    """Build a mapping from team display names to ESPN team IDs.

    Args:
        espn_teams_df: DataFrame with columns 'team_id' and 'team_name'

    Returns:
        dict mapping normalized team names to ESPN team IDs
    """
    mapping = {}
    for _, row in espn_teams_df.iterrows():
        tid = str(row["team_id"])
        name = str(row["team_name"])
        # Map both raw and normalized names
        mapping[name.lower()] = tid
        mapping[_normalize_name(name)] = tid

    return mapping


def merge_barttorvik_with_ids(
    barttorvik_df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Add team_id column to Barttorvik data by matching team names.

    Logs warnings for unmatched teams.
    """
    df = barttorvik_df.copy()

    def _find_id(name: str) -> str | None:
        # Try exact match first
        lower = name.lower()
        if lower in mapping:
            return mapping[lower]

        # Try manual override
        if name in MANUAL_OVERRIDES:
            override = MANUAL_OVERRIDES[name].lower()
            if override in mapping:
                return mapping[override]

        # Try normalized match
        normalized = _normalize_name(name)
        if normalized in mapping:
            return mapping[normalized]

        return None

    df["team_id"] = df["team_name"].apply(_find_id)

    unmatched = df[df["team_id"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Could not match %d Barttorvik teams: %s",
            len(unmatched),
            unmatched["team_name"].tolist()[:10],
        )

    return df
