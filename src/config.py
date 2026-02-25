"""Configuration loader for March Madness bracket prediction."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ESPNConfig:
    base_url: str
    scoreboard_url: str
    teams_url: str
    rankings_url: str
    rate_limit_seconds: float = 0.5
    max_retries: int = 3


@dataclass
class PoolConfig:
    pool_size: int = 285
    strategy: str = "leverage"
    champion_filter: bool = True
    luck_fade_threshold: float = 3.0
    early_round_chalk_threshold: float = 0.55
    upset_leverage_threshold: float = 2.0


@dataclass
class Config:
    espn: ESPNConfig
    seasons: list[int]
    power_conferences: list[str]
    round_points: dict[int, int]
    raw_data_path: Path
    processed_data_path: Path
    tournament_data_path: Path
    models_path: Path
    external_data_path: Path = field(default_factory=lambda: Path("data/external"))
    pool: PoolConfig = field(default_factory=PoolConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        config_path = Path(config_path)

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        project_root = config_path.parent

        # Parse pool config with safe defaults
        pool_raw = raw.get("pool", {})
        pool = PoolConfig(
            pool_size=pool_raw.get("pool_size", 285),
            strategy=pool_raw.get("strategy", "leverage"),
            champion_filter=pool_raw.get("champion_filter", True),
            luck_fade_threshold=pool_raw.get("luck_fade_threshold", 3.0),
            early_round_chalk_threshold=pool_raw.get("early_round_chalk_threshold", 0.55),
            upset_leverage_threshold=pool_raw.get("upset_leverage_threshold", 2.0),
        )

        return cls(
            espn=ESPNConfig(**raw["espn"]),
            seasons=raw["seasons"],
            power_conferences=raw["power_conferences"],
            round_points={int(k): v for k, v in raw["scoring"]["round_points"].items()},
            raw_data_path=project_root / raw["paths"]["raw_data"],
            processed_data_path=project_root / raw["paths"]["processed_data"],
            tournament_data_path=project_root / raw["paths"]["tournament_data"],
            models_path=project_root / raw["paths"]["models"],
            external_data_path=project_root / raw["paths"].get("external_data", "data/external"),
            pool=pool,
        )
