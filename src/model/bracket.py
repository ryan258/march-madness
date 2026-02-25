"""Recursive EV-optimal bracket builder."""

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.predict import predict_matchup

logger = logging.getLogger(__name__)

# ESPN scoring: points per round
DEFAULT_ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

# How much to weight pool-relative EV vs raw EV by round
# Early rounds = mostly raw EV (chalk). Late rounds = mostly pool EV (contrarian).
ROUND_WEIGHTS = {1: 0.1, 2: 0.2, 3: 0.5, 4: 0.7, 5: 0.9, 6: 1.0}


class BracketTeam:
    """A team in the bracket with its features and probability of reaching each round."""

    def __init__(
        self,
        team_id: str,
        name: str,
        seed: int,
        features: dict,
        public_pick_pct: dict[int, float] | None = None,
        champion_viable: bool = True,
    ):
        self.team_id = team_id
        self.name = name
        self.seed = seed
        self.features = features
        # P(reaching round r) - initialized: 1.0 for round 1
        self.reach_prob = {1: 1.0}
        # Per-round public pick percentages (default empty = no leverage)
        self.public_pick_pct = public_pick_pct or {}
        # Whether team meets AdjO/AdjD threshold for championship
        self.champion_viable = champion_viable

    def __repr__(self):
        return f"({self.seed}) {self.name}"


class BracketBuilder:
    """Build an EV-optimal bracket via recursive simulation."""

    def __init__(self, model, round_points: dict[int, int] | None = None):
        self.model = model
        self.round_points = round_points or DEFAULT_ROUND_POINTS

    def build_bracket(self, teams: list[BracketTeam]) -> list[dict]:
        """Build a full bracket from 64 teams seeded in bracket order.

        Teams should be ordered by matchup: [1v16 game1, 1v16 game2, ...]
        i.e., teams[0] vs teams[1] is the first game, etc.

        Returns list of picks per round with EV information.
        """
        if len(teams) not in (64, 32, 16, 8, 4, 2):
            raise ValueError(f"Expected power-of-2 teams, got {len(teams)}")

        all_picks = []
        current_round_teams = teams
        round_num = 1

        while len(current_round_teams) > 1:
            next_round_teams = []
            round_picks = []

            for i in range(0, len(current_round_teams), 2):
                t1 = current_round_teams[i]
                t2 = current_round_teams[i + 1]

                # Win probability for t1
                win_prob = predict_matchup(
                    self.model,
                    {**t1.features, "seed": t1.seed},
                    {**t2.features, "seed": t2.seed},
                )

                # EV for picking each team
                points = self.round_points.get(round_num, 10)
                ev_t1 = t1.reach_prob.get(round_num, 1.0) * win_prob * points
                ev_t2 = t2.reach_prob.get(round_num, 1.0) * (1 - win_prob) * points

                # Pick the team with higher EV
                if ev_t1 >= ev_t2:
                    winner = t1
                    winner_prob = win_prob
                    winner_ev = ev_t1
                else:
                    winner = t2
                    winner_prob = 1 - win_prob
                    winner_ev = ev_t2

                # Update reach probabilities for next round
                winner.reach_prob[round_num + 1] = (
                    winner.reach_prob.get(round_num, 1.0) * winner_prob
                )

                next_round_teams.append(winner)
                round_picks.append({
                    "round": round_num,
                    "team_1": str(t1),
                    "team_2": str(t2),
                    "winner": str(winner),
                    "win_prob": winner_prob,
                    "ev": winner_ev,
                    "points": points,
                })

            all_picks.extend(round_picks)
            current_round_teams = next_round_teams
            round_num += 1

        return all_picks

    def export_picks_csv(self, picks: list[dict], output_path: str | Path) -> None:
        """Export bracket picks to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "team_1", "team_2", "winner", "win_prob", "ev", "points"],
            )
            writer.writeheader()
            writer.writerows(picks)

        logger.info("Exported %d picks to %s", len(picks), output_path)


class LeverageBracketBuilder:
    """Build a pool-optimized bracket using leverage scoring.

    Maximizes relative edge over the field by considering public pick
    percentages, pool size, and contrarian value in late rounds.
    """

    def __init__(
        self,
        model,
        pool_size: int = 285,
        round_points: dict[int, int] | None = None,
        champion_filter: bool = True,
        luck_fade_threshold: float = 3.0,
    ):
        self.model = model
        self.pool_size = pool_size
        self.round_points = round_points or DEFAULT_ROUND_POINTS
        self.champion_filter = champion_filter
        self.luck_fade_threshold = luck_fade_threshold

    def _get_public_pct(self, team: BracketTeam, round_num: int) -> float:
        """Get public pick percentage for a team in a given round."""
        return team.public_pick_pct.get(round_num, 50.0)

    def _apply_luck_fade(self, win_prob: float, team: BracketTeam) -> float:
        """Reduce win probability for teams with high luck rating."""
        luck = team.features.get("luck", 0)
        if abs(luck) > self.luck_fade_threshold:
            return win_prob * 0.85
        return win_prob

    def _compute_leverage(self, win_prob: float, public_pct: float) -> float:
        """Compute leverage: model edge relative to public ownership."""
        return win_prob / max(public_pct / 100.0, 0.01)

    def _compute_pool_ev(
        self, reach_prob: float, win_prob: float, points: int, public_pct: float,
    ) -> float:
        """Compute pool-relative expected value.

        Accounts for expected number of other entrants sharing this pick.
        """
        expected_sharers = self.pool_size * (public_pct / 100.0)
        return reach_prob * win_prob * points / max(expected_sharers, 1.0)

    def _blend_ev(
        self, raw_ev: float, pool_ev: float, round_num: int,
    ) -> float:
        """Blend raw EV and pool EV based on round weight."""
        w = ROUND_WEIGHTS.get(round_num, 0.5)
        return (1 - w) * raw_ev + w * pool_ev

    def _pick_reason(
        self, winner: BracketTeam, loser: BracketTeam,
        round_num: int, leverage: float, is_upset: bool,
    ) -> str:
        """Generate human-readable pick reason."""
        if round_num <= 2 and not is_upset:
            return "chalk"
        if round_num >= 5:
            return f"contrarian (leverage {leverage:.1f}x)"
        if is_upset and leverage > 1.5:
            return f"value upset (leverage {leverage:.1f}x)"
        if leverage > 2.0:
            return f"high leverage ({leverage:.1f}x)"
        return "model favored"

    def build_bracket(self, teams: list[BracketTeam]) -> list[dict]:
        """Build a leverage-optimized bracket.

        Same loop structure as BracketBuilder but incorporates:
        - Luck fade for overperforming teams
        - Leverage scoring (win_prob / public_pct)
        - Pool-relative EV (points shared among pool)
        - Blended EV weighting by round
        - 12-vs-5 upset exception
        - Champion viability filter in round 6
        """
        if len(teams) not in (64, 32, 16, 8, 4, 2):
            raise ValueError(f"Expected power-of-2 teams, got {len(teams)}")

        all_picks = []
        current_round_teams = teams
        round_num = 1

        while len(current_round_teams) > 1:
            next_round_teams = []
            round_picks = []

            for i in range(0, len(current_round_teams), 2):
                t1 = current_round_teams[i]
                t2 = current_round_teams[i + 1]

                # Base win probability for t1
                raw_win_prob = predict_matchup(
                    self.model,
                    {**t1.features, "seed": t1.seed},
                    {**t2.features, "seed": t2.seed},
                )

                # Apply luck fade
                win_prob_t1 = self._apply_luck_fade(raw_win_prob, t1)
                win_prob_t2 = self._apply_luck_fade(1 - raw_win_prob, t2)
                # Renormalize after independent fading
                total = win_prob_t1 + win_prob_t2
                if total > 0:
                    win_prob_t1 /= total
                    win_prob_t2 /= total

                points = self.round_points.get(round_num, 10)
                reach_t1 = t1.reach_prob.get(round_num, 1.0)
                reach_t2 = t2.reach_prob.get(round_num, 1.0)

                # Public pick percentages
                pub_t1 = self._get_public_pct(t1, round_num)
                pub_t2 = self._get_public_pct(t2, round_num)

                # Leverage scores
                leverage_t1 = self._compute_leverage(win_prob_t1, pub_t1)
                leverage_t2 = self._compute_leverage(win_prob_t2, pub_t2)

                # Raw EV
                raw_ev_t1 = reach_t1 * win_prob_t1 * points
                raw_ev_t2 = reach_t2 * win_prob_t2 * points

                # Pool-relative EV
                pool_ev_t1 = self._compute_pool_ev(reach_t1, win_prob_t1, points, pub_t1)
                pool_ev_t2 = self._compute_pool_ev(reach_t2, win_prob_t2, points, pub_t2)

                # Blended EV
                blended_t1 = self._blend_ev(raw_ev_t1, pool_ev_t1, round_num)
                blended_t2 = self._blend_ev(raw_ev_t2, pool_ev_t2, round_num)

                # 12-vs-5 exception in round 1
                if round_num == 1:
                    if t1.seed == 12 and t2.seed == 5:
                        if win_prob_t1 > 0.45 and leverage_t1 > 1.5:
                            blended_t1 = blended_t2 + 1  # Force upset pick
                    elif t2.seed == 12 and t1.seed == 5:
                        if win_prob_t2 > 0.45 and leverage_t2 > 1.5:
                            blended_t2 = blended_t1 + 1  # Force upset pick

                # Champion filter in round 6
                if round_num == 6 and self.champion_filter:
                    t1_viable = t1.champion_viable
                    t2_viable = t2.champion_viable
                    if t1_viable and not t2_viable:
                        blended_t1 = blended_t2 + 1
                    elif t2_viable and not t1_viable:
                        blended_t2 = blended_t1 + 1

                # Pick winner based on blended EV
                if blended_t1 >= blended_t2:
                    winner = t1
                    loser = t2
                    winner_prob = win_prob_t1
                    winner_ev = blended_t1
                    winner_leverage = leverage_t1
                    winner_pub = pub_t1
                    winner_pool_ev = pool_ev_t1
                    is_upset = t1.seed > t2.seed
                else:
                    winner = t2
                    loser = t1
                    winner_prob = win_prob_t2
                    winner_ev = blended_t2
                    winner_leverage = leverage_t2
                    winner_pub = pub_t2
                    winner_pool_ev = pool_ev_t2
                    is_upset = t2.seed > t1.seed

                # Update reach probabilities for next round
                winner.reach_prob[round_num + 1] = (
                    winner.reach_prob.get(round_num, 1.0) * winner_prob
                )

                reason = self._pick_reason(
                    winner, loser, round_num, winner_leverage, is_upset,
                )

                next_round_teams.append(winner)
                round_picks.append({
                    "round": round_num,
                    "team_1": str(t1),
                    "team_2": str(t2),
                    "winner": str(winner),
                    "win_prob": winner_prob,
                    "ev": winner_ev,
                    "points": points,
                    "leverage": winner_leverage,
                    "public_pct": winner_pub,
                    "pool_ev": winner_pool_ev,
                    "pick_reason": reason,
                })

            all_picks.extend(round_picks)
            current_round_teams = next_round_teams
            round_num += 1

        return all_picks

    def export_picks_csv(self, picks: list[dict], output_path: str | Path) -> None:
        """Export leverage bracket picks to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "round", "team_1", "team_2", "winner", "win_prob", "ev",
            "points", "leverage", "public_pct", "pool_ev", "pick_reason",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(picks)

        logger.info("Exported %d leverage picks to %s", len(picks), output_path)


def build_bracket_from_features(
    model,
    features: pd.DataFrame,
    bracket_teams: list[dict],
    round_points: dict[int, int] | None = None,
) -> list[dict]:
    """Convenience function to build a bracket from features DataFrame.

    bracket_teams: list of dicts with keys: team_id, name, seed
    (ordered by bracket position)
    """
    teams = []
    for team_info in bracket_teams:
        tid = str(team_info["team_id"])
        team_feats = features[features["team_id"] == tid]
        if team_feats.empty:
            feat_dict = {}
        else:
            feat_dict = team_feats.iloc[-1].to_dict()

        teams.append(BracketTeam(
            team_id=tid,
            name=team_info["name"],
            seed=team_info["seed"],
            features=feat_dict,
        ))

    builder = BracketBuilder(model, round_points)
    return builder.build_bracket(teams)


def build_leverage_bracket_from_features(
    model,
    features: pd.DataFrame,
    bracket_teams: list[dict],
    pool_size: int = 285,
    round_points: dict[int, int] | None = None,
    champion_filter: bool = True,
    luck_fade_threshold: float = 3.0,
    public_picks_csv: pd.DataFrame | None = None,
) -> list[dict]:
    """Convenience function to build a leverage bracket from features DataFrame.

    bracket_teams: list of dicts with keys: team_id, name, seed
    (ordered by bracket position). Optionally: public_pick_pct, champion_viable.
    """
    from src.data.public_picks import PublicPicksLoader

    teams = []
    for team_info in bracket_teams:
        tid = str(team_info["team_id"])
        team_feats = features[features["team_id"] == tid]
        if team_feats.empty:
            feat_dict = {}
        else:
            feat_dict = team_feats.iloc[-1].to_dict()

        seed = team_info["seed"]
        name = team_info.get("name", tid)

        # Get public pick percentages
        public_pct = team_info.get("public_pick_pct")
        if public_pct is None:
            public_pct = PublicPicksLoader.get_pick_pct_for_team(
                seed, name, public_picks_csv,
            )

        # Get champion viability
        champion_viable = team_info.get("champion_viable")
        if champion_viable is None:
            champion_viable = bool(feat_dict.get("champion_viable", True))

        teams.append(BracketTeam(
            team_id=tid,
            name=name,
            seed=seed,
            features=feat_dict,
            public_pick_pct=public_pct,
            champion_viable=champion_viable,
        ))

    builder = LeverageBracketBuilder(
        model,
        pool_size=pool_size,
        round_points=round_points,
        champion_filter=champion_filter,
        luck_fade_threshold=luck_fade_threshold,
    )
    return builder.build_bracket(teams)
