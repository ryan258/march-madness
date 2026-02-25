"""March Madness Bracket Prediction - Streamlit Frontend."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data.storage import data_exists, load_parquet
from src.model.bracket import BracketBuilder, BracketTeam, LeverageBracketBuilder
from src.model.dataset import DIFF_COLS
from src.model.predict import predict_matchup
from src.model.tiebreaker import predict_championship_total

st.set_page_config(page_title="March Madness Predictor", layout="wide")


@st.cache_data
def load_config():
    return Config.load(PROJECT_ROOT / "config.yaml")


@st.cache_data
def load_features():
    config = load_config()
    path = config.processed_data_path / "features.parquet"
    if data_exists(path):
        return load_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_tournament_results():
    config = load_config()
    path = config.tournament_data_path / "tournament_results.parquet"
    if data_exists(path):
        return load_parquet(path)
    return pd.DataFrame()


@st.cache_resource
def load_model(model_name: str = "bracket_model.joblib"):
    config = load_config()
    model_path = config.models_path / model_name
    if model_path.exists():
        return joblib.load(model_path)
    return None


def main():
    st.title("March Madness Bracket Predictor")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        features = load_features()
        if features.empty:
            st.warning("No feature data found. Run the data pipeline first.")
            st.code(
                "python -c \"\n"
                "from src.config import Config\n"
                "from src.features.builder import build_features\n"
                "config = Config.load()\n"
                "build_features(config)\n"
                "\"",
                language="bash",
            )

        available_seasons = sorted(features["season"].unique()) if not features.empty else []
        selected_season = st.selectbox(
            "Season", available_seasons, index=len(available_seasons) - 1 if available_seasons else 0
        )

        model_files = list(Path(load_config().models_path).glob("*.joblib"))
        model_names = [f.name for f in model_files] if model_files else ["bracket_model.joblib"]
        selected_model = st.selectbox("Model", model_names)

        model = load_model(selected_model)
        if model is None:
            st.warning("No trained model found. Train a model first.")

        # Pool optimizer settings
        st.header("Pool Settings")
        config = load_config()
        pool_size = st.slider("Pool Size", 10, 500, config.pool.pool_size)
        strategy = st.selectbox("Strategy", ["Pure EV", "Leverage", "Hybrid"], index=1)
        champion_filter = st.checkbox("Champion Filter", value=config.pool.champion_filter)
        luck_fade = st.slider(
            "Luck Fade Threshold", 0.0, 10.0, config.pool.luck_fade_threshold, 0.5,
        )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bracket View", "Team Stats", "Matchup Explorer", "Model Performance",
        "Pool Optimizer",
    ])

    # Tab 1: Bracket View
    with tab1:
        render_bracket_tab(features, model, selected_season)

    # Tab 2: Team Stats
    with tab2:
        render_team_stats_tab(features, selected_season)

    # Tab 3: Matchup Explorer
    with tab3:
        render_matchup_tab(features, model, selected_season)

    # Tab 4: Model Performance
    with tab4:
        render_model_performance_tab()

    # Tab 5: Pool Optimizer
    with tab5:
        render_pool_optimizer_tab(
            features, model, selected_season,
            pool_size, strategy, champion_filter, luck_fade,
        )


def render_bracket_tab(features: pd.DataFrame, model, selected_season: int):
    """Bracket view with color-coded win probabilities."""
    st.header("Bracket View")

    if features.empty or model is None:
        st.info("Load features and train a model to view bracket predictions.")
        return

    season_features = features[features["season"] == selected_season]
    if season_features.empty:
        st.warning(f"No features found for season {selected_season}")
        return

    # Load tournament results for this season to get bracket teams
    tournament = load_tournament_results()
    if tournament.empty:
        st.info("No tournament data available. Enter teams manually below.")
        _manual_bracket_input(season_features, model)
        return

    season_tourney = tournament[tournament["season"] == selected_season]
    if season_tourney.empty:
        st.info(f"No tournament data for {selected_season}. Enter teams manually.")
        _manual_bracket_input(season_features, model)
        return

    # Extract unique teams from tournament
    teams_1 = season_tourney[["team_1_id", "team_1_name", "seed_1"]].rename(
        columns={"team_1_id": "team_id", "team_1_name": "name", "seed_1": "seed"}
    )
    teams_2 = season_tourney[["team_2_id", "team_2_name", "seed_2"]].rename(
        columns={"team_2_id": "team_id", "team_2_name": "name", "seed_2": "seed"}
    )
    all_teams = pd.concat([teams_1, teams_2]).drop_duplicates(subset="team_id")
    all_teams["team_id"] = all_teams["team_id"].astype(str)
    all_teams = all_teams.sort_values("seed")

    st.write(f"**{len(all_teams)} tournament teams for {selected_season}**")

    # Build bracket teams list
    bracket_teams = []
    for _, row in all_teams.iterrows():
        tid = str(row["team_id"])
        t_feats = season_features[season_features["team_id"] == tid]
        feat_dict = t_feats.iloc[-1].to_dict() if not t_feats.empty else {}
        bracket_teams.append(BracketTeam(
            team_id=tid, name=row["name"], seed=int(row["seed"]), features=feat_dict
        ))

    # Pad to nearest power of 2 if needed
    target_size = 2 ** int(np.ceil(np.log2(max(len(bracket_teams), 2))))
    while len(bracket_teams) < target_size:
        bracket_teams.append(BracketTeam("0", "BYE", 16, {}))

    if st.button("Generate Bracket"):
        builder = BracketBuilder(model, load_config().round_points)
        picks = builder.build_bracket(bracket_teams)

        round_names = {
            1: "First Round", 2: "Second Round", 3: "Sweet 16",
            4: "Elite Eight", 5: "Final Four", 6: "Championship",
        }

        for round_num in sorted(set(p["round"] for p in picks)):
            st.subheader(round_names.get(round_num, f"Round {round_num}"))
            round_picks = [p for p in picks if p["round"] == round_num]

            for pick in round_picks:
                prob = pick["win_prob"]
                # Color: green for confident (>0.7), yellow (0.5-0.7), red (<0.5 = upset)
                if prob > 0.7:
                    color = "🟢"
                elif prob > 0.5:
                    color = "🟡"
                else:
                    color = "🔴"

                st.write(
                    f"{color} **{pick['winner']}** beats "
                    f"{pick['team_1'] if pick['winner'] != pick['team_1'] else pick['team_2']} "
                    f"({prob:.1%} | EV: {pick['ev']:.1f})"
                )


def _manual_bracket_input(features: pd.DataFrame, model):
    """Allow manual team entry for bracket generation."""
    st.write("Enter team IDs (comma-separated) to generate matchup predictions:")
    team_input = st.text_input("Team IDs", placeholder="e.g., 2250,150,2305")
    if team_input and model:
        team_ids = [t.strip() for t in team_input.split(",")]
        season_features = features[features["team_id"].isin(team_ids)]
        if not season_features.empty:
            for i in range(0, len(team_ids) - 1, 2):
                t1_feats = season_features[season_features["team_id"] == team_ids[i]]
                t2_feats = season_features[season_features["team_id"] == team_ids[i + 1]]
                if not t1_feats.empty and not t2_feats.empty:
                    prob = predict_matchup(model, t1_feats.iloc[-1].to_dict(), t2_feats.iloc[-1].to_dict())
                    st.write(f"Team {team_ids[i]} vs Team {team_ids[i+1]}: {prob:.1%}")


def render_team_stats_tab(features: pd.DataFrame, selected_season: int):
    """Searchable table of team features."""
    st.header("Team Stats")

    if features.empty:
        st.info("No feature data available.")
        return

    season_data = features[features["season"] == selected_season].copy()
    if season_data.empty:
        st.warning(f"No data for season {selected_season}")
        return

    # Search
    search = st.text_input("Search by team ID", "")
    if search:
        season_data = season_data[
            season_data["team_id"].astype(str).str.contains(search, case=False)
        ]

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        ["efficiency_ratio", "momentum", "sos", "upset_propensity", "off_efficiency", "def_efficiency"],
    )
    sort_asc = st.checkbox("Ascending", value=False)

    display_cols = [
        "team_id", "efficiency_ratio", "off_efficiency", "def_efficiency",
        "momentum", "sos", "upset_propensity",
    ]
    available_cols = [c for c in display_cols if c in season_data.columns]
    sorted_data = season_data[available_cols].sort_values(sort_col, ascending=sort_asc)

    st.dataframe(sorted_data, use_container_width=True, hide_index=True)
    st.write(f"Showing {len(sorted_data)} teams")

    # Feature distribution charts
    if st.checkbox("Show feature distributions"):
        for col in ["efficiency_ratio", "momentum", "sos"]:
            if col in season_data.columns:
                fig = px.histogram(season_data, x=col, nbins=30, title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)


def render_matchup_tab(features: pd.DataFrame, model, selected_season: int):
    """Head-to-head matchup explorer."""
    st.header("Matchup Explorer")

    if features.empty or model is None:
        st.info("Load features and train a model to explore matchups.")
        return

    season_data = features[features["season"] == selected_season]
    if season_data.empty:
        st.warning(f"No data for season {selected_season}")
        return

    team_ids = sorted(season_data["team_id"].unique())

    col1, col2 = st.columns(2)
    with col1:
        team_1 = st.selectbox("Team 1", team_ids, key="t1")
    with col2:
        team_2 = st.selectbox("Team 2", team_ids, index=min(1, len(team_ids) - 1), key="t2")

    if team_1 == team_2:
        st.warning("Select two different teams.")
        return

    t1_feats = season_data[season_data["team_id"] == team_1].iloc[-1]
    t2_feats = season_data[season_data["team_id"] == team_2].iloc[-1]

    prob = predict_matchup(model, t1_feats.to_dict(), t2_feats.to_dict())

    # Probability bar
    st.subheader("Win Probability")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Team {team_1}", f"{prob:.1%}")
    with col2:
        st.metric(f"Team {team_2}", f"{1 - prob:.1%}")

    st.progress(prob)

    # Feature comparison
    st.subheader("Feature Comparison")
    compare_cols = [
        "efficiency_ratio", "off_efficiency", "def_efficiency",
        "momentum", "sos", "upset_propensity",
    ]
    available = [c for c in compare_cols if c in t1_feats.index]

    comparison = pd.DataFrame({
        "Feature": available,
        f"Team {team_1}": [t1_feats[c] for c in available],
        f"Team {team_2}": [t2_feats[c] for c in available],
    })
    comparison["Difference"] = comparison[f"Team {team_1}"] - comparison[f"Team {team_2}"]
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Radar chart
    if available:
        # Normalize for radar
        t1_vals = [float(t1_feats[c]) for c in available]
        t2_vals = [float(t2_feats[c]) for c in available]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=t1_vals, theta=available, fill="toself", name=f"Team {team_1}"
        ))
        fig.add_trace(go.Scatterpolar(
            r=t2_vals, theta=available, fill="toself", name=f"Team {team_2}"
        ))
        fig.update_layout(title="Feature Comparison Radar", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)


def render_model_performance_tab():
    """Display model performance metrics and calibration curve."""
    st.header("Model Performance")

    config = load_config()
    calibration_path = config.models_path / "calibration.png"

    # Show calibration curve if available
    if calibration_path.exists():
        st.subheader("Calibration Curve")
        st.image(str(calibration_path))
    else:
        st.info("No calibration plot found. Run model evaluation first.")
        st.code(
            "python -c \"\n"
            "from src.config import Config\n"
            "from src.model.evaluate import leave_one_season_out_cv, plot_calibration\n"
            "from src.model.dataset import build_matchup_dataset\n"
            "from src.data.storage import load_parquet\n"
            "config = Config.load()\n"
            "features = load_parquet(config.processed_data_path / 'features.parquet')\n"
            "tournament = load_parquet(config.tournament_data_path / 'tournament_results.parquet')\n"
            "dataset = build_matchup_dataset(tournament, features)\n"
            "results = leave_one_season_out_cv(dataset)\n"
            "plot_calibration(results['y_true'], results['y_prob'], config.models_path / 'calibration.png')\n"
            "\"",
            language="bash",
        )

    # Show metrics summary
    st.subheader("Expected Performance Targets")
    targets = pd.DataFrame({
        "Metric": ["Accuracy", "Log Loss", "Brier Score", "AUC"],
        "Target": [">65%", "<0.65", "<0.23", ">0.68"],
        "Baseline (seed-based)": ["~64%", "~0.66", "~0.24", "~0.65"],
    })
    st.dataframe(targets, use_container_width=True, hide_index=True)

    # Feature importance (if model is loaded)
    model = load_model()
    if model is not None:
        st.subheader("Feature Importance")
        try:
            # For LogReg pipeline
            if hasattr(model, "named_steps"):
                clf = model.named_steps["clf"]
                if hasattr(clf, "coef_"):
                    coefs = clf.coef_[0]
                    importance = pd.DataFrame({
                        "Feature": DIFF_COLS,
                        "Coefficient": coefs,
                        "Abs Coefficient": np.abs(coefs),
                    }).sort_values("Abs Coefficient", ascending=False)

                    fig = px.bar(
                        importance,
                        x="Coefficient",
                        y="Feature",
                        orientation="h",
                        title="Logistic Regression Coefficients",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display feature importance: {e}")


def render_pool_optimizer_tab(
    features: pd.DataFrame,
    model,
    selected_season: int,
    pool_size: int,
    strategy: str,
    champion_filter: bool,
    luck_fade: float,
):
    """Pool Optimizer tab for game-theory bracket optimization."""
    st.header("Pool Optimizer")

    if features.empty or model is None:
        st.info("Load features and train a model to use the pool optimizer.")
        return

    season_features = features[features["season"] == selected_season]
    if season_features.empty:
        st.warning(f"No features for season {selected_season}")
        return

    # Data loading section
    col_left, col_right = st.columns(2)

    barttorvik_df = pd.DataFrame()
    public_picks_df = pd.DataFrame()

    with col_left:
        st.subheader("Barttorvik Data")
        bart_upload = st.file_uploader("Upload Barttorvik CSV", type="csv", key="bart_csv")
        if bart_upload is not None:
            from src.data.barttorvik_client import BarttovikClient
            barttorvik_df = BarttovikClient.load_from_csv(bart_upload)
            st.success(f"Loaded {len(barttorvik_df)} teams from CSV")
        else:
            config = load_config()
            bart_path = config.external_data_path / f"barttorvik_{selected_season}.parquet"
            if data_exists(bart_path):
                barttorvik_df = load_parquet(bart_path)
                st.info(f"Using cached Barttorvik data ({len(barttorvik_df)} teams)")
            else:
                if st.button("Fetch from Barttorvik"):
                    from src.data.barttorvik_client import BarttovikClient
                    with st.spinner("Fetching Barttorvik rankings..."):
                        client = BarttovikClient()
                        barttorvik_df = client.fetch_and_cache(
                            selected_season, config.external_data_path,
                        )
                    if not barttorvik_df.empty:
                        st.success(f"Fetched {len(barttorvik_df)} teams")
                    else:
                        st.error("Failed to fetch data. Upload CSV instead.")

    with col_right:
        st.subheader("Public Picks Data")
        picks_upload = st.file_uploader("Upload Public Picks CSV", type="csv", key="picks_csv")
        if picks_upload is not None:
            from src.data.public_picks import PublicPicksLoader
            public_picks_df = PublicPicksLoader.load_from_csv(picks_upload)
            st.success(f"Loaded picks for {len(public_picks_df)} teams")
        else:
            if st.button("Use Seed Defaults"):
                st.info("Using historical seed-based defaults for public picks")

    # Champion viable teams table
    if not barttorvik_df.empty and "adj_o" in barttorvik_df.columns:
        st.subheader("Champion Viable Teams")
        from src.features.advanced_metrics import compute_advanced_metrics
        if "team_id" in barttorvik_df.columns:
            adv = compute_advanced_metrics(barttorvik_df)
            viable = adv[adv["champion_viable"]]
            if not viable.empty:
                display_viable = viable[["team_id", "adj_o", "adj_d", "adj_tempo"]].copy()
                display_viable = display_viable.sort_values("adj_o", ascending=False)
                st.dataframe(display_viable, use_container_width=True, hide_index=True)
                st.write(f"{len(viable)} teams meet Top 20 AdjO + AdjD threshold")
            else:
                st.info("No teams computed as champion viable yet.")

    # Leverage analysis table
    if not season_features.empty:
        st.subheader("Leverage Analysis")
        from src.data.public_picks import PublicPicksLoader

        leverage_rows = []
        for _, row in season_features.iterrows():
            tid = str(row["team_id"])
            seed = int(row.get("seed", 8)) if "seed" in row.index else 8
            pcts = PublicPicksLoader.get_pick_pct_for_team(
                seed, None, public_picks_df if not public_picks_df.empty else None,
            )
            # Use round 6 (champion) public pick %
            champ_pct = pcts.get(6, 1.0)
            eff = row.get("efficiency_ratio", 0)
            leverage_rows.append({
                "team_id": tid,
                "efficiency": round(eff, 3),
                "public_champ_%": round(champ_pct, 2),
                "leverage_r6": round(eff / max(champ_pct / 100, 0.01), 2) if eff > 0 else 0,
            })

        if leverage_rows:
            lev_df = pd.DataFrame(leverage_rows).sort_values("leverage_r6", ascending=False)
            st.dataframe(lev_df.head(20), use_container_width=True, hide_index=True)

    # Generate bracket
    st.subheader("Generate Optimized Bracket")

    tournament = load_tournament_results()
    season_tourney = tournament[tournament["season"] == selected_season] if not tournament.empty else pd.DataFrame()

    if season_tourney.empty:
        st.info("No tournament data for this season. Cannot generate bracket.")
        return

    # Extract teams
    teams_1 = season_tourney[["team_1_id", "team_1_name", "seed_1"]].rename(
        columns={"team_1_id": "team_id", "team_1_name": "name", "seed_1": "seed"}
    )
    teams_2 = season_tourney[["team_2_id", "team_2_name", "seed_2"]].rename(
        columns={"team_2_id": "team_id", "team_2_name": "name", "seed_2": "seed"}
    )
    all_teams = pd.concat([teams_1, teams_2]).drop_duplicates(subset="team_id")
    all_teams["team_id"] = all_teams["team_id"].astype(str)
    all_teams = all_teams.sort_values("seed")

    compare_mode = st.checkbox("Compare Pure EV vs Leverage")

    if st.button("Generate Pool-Optimized Bracket", key="gen_leverage"):
        from src.data.public_picks import PublicPicksLoader

        bracket_teams_list = []
        for _, row in all_teams.iterrows():
            tid = str(row["team_id"])
            t_feats = season_features[season_features["team_id"] == tid]
            feat_dict = t_feats.iloc[-1].to_dict() if not t_feats.empty else {}
            seed = int(row["seed"])
            name = row["name"]

            pct = PublicPicksLoader.get_pick_pct_for_team(
                seed, name, public_picks_df if not public_picks_df.empty else None,
            )
            viable = bool(feat_dict.get("champion_viable", True))

            bracket_teams_list.append(BracketTeam(
                team_id=tid, name=name, seed=seed, features=feat_dict,
                public_pick_pct=pct, champion_viable=viable,
            ))

        # Pad to power of 2
        target_size = 2 ** int(np.ceil(np.log2(max(len(bracket_teams_list), 2))))
        while len(bracket_teams_list) < target_size:
            bracket_teams_list.append(BracketTeam("0", "BYE", 16, {}))

        config = load_config()

        if strategy == "Pure EV":
            builder = BracketBuilder(model, config.round_points)
        else:
            builder = LeverageBracketBuilder(
                model,
                pool_size=pool_size,
                round_points=config.round_points,
                champion_filter=champion_filter,
                luck_fade_threshold=luck_fade,
            )

        picks = builder.build_bracket(bracket_teams_list)

        round_names = {
            1: "First Round", 2: "Second Round", 3: "Sweet 16",
            4: "Elite Eight", 5: "Final Four", 6: "Championship",
        }

        for round_num in sorted(set(p["round"] for p in picks)):
            st.subheader(round_names.get(round_num, f"Round {round_num}"))
            round_picks = [p for p in picks if p["round"] == round_num]

            for pick in round_picks:
                prob = pick["win_prob"]
                if prob > 0.7:
                    color = "🟢"
                elif prob > 0.5:
                    color = "🟡"
                else:
                    color = "🔴"

                reason = pick.get("pick_reason", "")
                leverage = pick.get("leverage", 0)
                reason_str = f" | {reason}" if reason else ""
                lev_str = f" | Lev: {leverage:.1f}x" if leverage else ""

                st.write(
                    f"{color} **{pick['winner']}** beats "
                    f"{pick['team_1'] if pick['winner'] != pick['team_1'] else pick['team_2']} "
                    f"({prob:.1%} | EV: {pick['ev']:.1f}{lev_str}{reason_str})"
                )

        # Tiebreaker
        champ_pick = [p for p in picks if p["round"] == max(p["round"] for p in picks)]
        if champ_pick:
            st.subheader("Tiebreaker Prediction")
            # Find the two finalists' features
            final_teams = [p for p in picks if p["round"] == max(p["round"] for p in picks) - 1]
            if len(final_teams) >= 2:
                # Get features for finalists
                f1 = {}
                f2 = {}
                for bt in bracket_teams_list:
                    if str(bt) == final_teams[0].get("winner", ""):
                        f1 = bt.features
                    if str(bt) == final_teams[1].get("winner", ""):
                        f2 = bt.features
                total = predict_championship_total(f1, f2)
                st.metric("Predicted Championship Total", f"{total} points")

        # Compare mode
        if compare_mode and strategy != "Pure EV":
            st.subheader("Pure EV vs Leverage Comparison")
            # Rebuild teams for pure EV (need fresh reach_prob)
            ev_teams = []
            for _, row in all_teams.iterrows():
                tid = str(row["team_id"])
                t_feats = season_features[season_features["team_id"] == tid]
                feat_dict = t_feats.iloc[-1].to_dict() if not t_feats.empty else {}
                ev_teams.append(BracketTeam(
                    team_id=tid, name=row["name"], seed=int(row["seed"]),
                    features=feat_dict,
                ))
            target_size = 2 ** int(np.ceil(np.log2(max(len(ev_teams), 2))))
            while len(ev_teams) < target_size:
                ev_teams.append(BracketTeam("0", "BYE", 16, {}))

            ev_builder = BracketBuilder(model, config.round_points)
            ev_picks = ev_builder.build_bracket(ev_teams)

            # Find divergences
            divergences = []
            for lp, ep in zip(picks, ev_picks):
                if lp["winner"] != ep["winner"]:
                    divergences.append({
                        "Round": lp["round"],
                        "Matchup": f"{lp['team_1']} vs {lp['team_2']}",
                        "Pure EV Pick": ep["winner"],
                        "Leverage Pick": lp["winner"],
                        "Leverage Reason": lp.get("pick_reason", ""),
                    })

            if divergences:
                st.dataframe(
                    pd.DataFrame(divergences),
                    use_container_width=True, hide_index=True,
                )
                st.write(f"**{len(divergences)} differences** between strategies")
            else:
                st.info("Both strategies produced identical brackets.")


if __name__ == "__main__":
    main()
