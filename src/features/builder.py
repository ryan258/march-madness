"""Feature builder: orchestrates all features into one table."""

import logging
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.sdv_loader import load_mbb_team_boxscores
from src.data.storage import data_exists, load_parquet, save_parquet
from src.features.efficiency import compute_efficiency
from src.features.momentum import compute_momentum
from src.features.strength_of_schedule import compute_strength_of_schedule
from src.features.upset_propensity import compute_upset_propensity
from src.features.advanced_metrics import compute_advanced_metrics

logger = logging.getLogger(__name__)


def build_features(config: Config, force: bool = False) -> pd.DataFrame:
    """Build feature table for all teams across all seasons.

    Orchestrates all four feature modules, merges on (team_id, season),
    imputes missing values with league median.

    Returns DataFrame with columns:
        team_id, season, efficiency_ratio, off_efficiency, def_efficiency,
        momentum, sos, upset_propensity, adj_o, adj_d, adj_tempo, luck,
        champion_viable
    """
    output_path = config.processed_data_path / "features.parquet"
    if not force and data_exists(output_path):
        logger.info("Loading cached features from %s", output_path)
        return load_parquet(output_path)

    # Load raw data
    logger.info("Loading team boxscores...")
    boxscores = load_mbb_team_boxscores(config.seasons, config.raw_data_path)

    if boxscores.empty:
        logger.warning("No boxscore data available")
        return pd.DataFrame()

    logger.info("Loaded %d boxscore rows", len(boxscores))

    # Compute each feature
    logger.info("Computing efficiency...")
    efficiency = compute_efficiency(boxscores)

    logger.info("Computing momentum...")
    momentum = compute_momentum(boxscores)

    logger.info("Computing strength of schedule...")
    sos = compute_strength_of_schedule(boxscores)

    logger.info("Computing upset propensity...")
    upset = compute_upset_propensity(
        boxscores,
        power_conferences=set(config.power_conferences),
    )

    # Merge all features
    features = efficiency
    for feat_df in [momentum, sos, upset]:
        if not feat_df.empty:
            # Ensure matching types for merge
            feat_df["team_id"] = feat_df["team_id"].astype(str)
            features["team_id"] = features["team_id"].astype(str)
            features = features.merge(
                feat_df,
                on=["team_id", "season"],
                how="outer",
            )

    # Integrate Barttorvik advanced metrics if available
    for season in config.seasons:
        bart_path = config.external_data_path / f"barttorvik_{season}.parquet"
        if data_exists(bart_path):
            logger.info("Loading Barttorvik data for season %d", season)
            bart_df = load_parquet(bart_path)
            if not bart_df.empty and "team_id" in bart_df.columns:
                adv = compute_advanced_metrics(bart_df)
                if not adv.empty:
                    adv["team_id"] = adv["team_id"].astype(str)
                    features["team_id"] = features["team_id"].astype(str)
                    features = features.merge(
                        adv,
                        on=["team_id", "season"],
                        how="outer",
                    )

    # Impute missing values with league median per season
    numeric_cols = [
        "efficiency_ratio", "off_efficiency", "def_efficiency",
        "momentum", "sos", "upset_propensity",
        "adj_o", "adj_d", "adj_tempo", "luck",
    ]
    for col in numeric_cols:
        if col in features.columns:
            season_median = features.groupby("season")[col].transform("median")
            features[col] = features[col].fillna(season_median)
            # If still NaN (entire season missing), fill with global median
            features[col] = features[col].fillna(features[col].median())

    # champion_viable passes through without imputation (default False)
    if "champion_viable" in features.columns:
        features["champion_viable"] = features["champion_viable"].fillna(False)

    # Fill any remaining NaN with 0
    features = features.fillna(0)

    save_parquet(features, output_path)
    logger.info("Saved features for %d team-seasons to %s", len(features), output_path)

    return features
