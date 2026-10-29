"""
model.py
--------
Trains the production NFL MVP model: a single calibrated logistic regression.

Why logistic regression and not XGBoost?
  With only one MVP per season there are ~26 positive examples in the modern
  (1999+) era and ~67 across all history. That is far too few for gradient
  boosting to learn reliable non-linear splits — in testing XGBoost (classifier
  and ranker) topped out around 42% top-1 on the modern era vs 61.5% for this
  logistic model. Linear models generalise better when positive examples are
  scarce, so logistic regression is the right production choice.

Era:
  By default we train on 1999+ ("modern" era) where advanced stats such as EPA
  exist and are the model's strongest predictors. Pre-1999 seasons lack EPA and
  dilute modern accuracy, so they are excluded by default. Pass --start 1957 to
  train on the full history (lower accuracy, but covers the earlier MVP era).

A position-aware variant (separate model per position group) lives in
model_position_aware.py as an experiment; it under-performs this model at the
current data size and is kept for reference / future work with more data.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = PROCESSED_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------

PASSING_FEATURES = [
    "pass_yds", "pass_td", "pass_int", "pass_rating", "pass_cmp_pct",
    "pass_ypa", "td_int_ratio", "pass_epa", "pacr", "dakota",
    "pass_yds_pg", "pass_td_pg", "pass_epa_pg",
    "pass_yds_league_zscore", "pass_td_league_zscore", "pass_epa_league_zscore",
    "pass_yds_league_rank", "pass_td_league_rank", "pass_epa_league_rank",
    "pass_epa_vs_league_avg",
    "pass_epa_qb_pos_zscore", "pass_rating_qb_pos_zscore",
    "pass_yds_qb_pos_pct", "pass_td_qb_pos_pct",
]

RUSHING_FEATURES = [
    "rush_yds", "rush_td", "rush_att", "rush_epa", "rush_ypc",
    "rush_yds_pg", "rush_td_pg",
    "rush_yds_league_zscore", "rush_td_league_zscore",
    "rush_yds_league_rank", "rush_yds_vs_league_avg",
]

RECEIVING_FEATURES = [
    "rec", "rec_yds", "rec_td", "targets", "target_share", "wopr", "racr",
    "rec_yds_pg", "rec_td_pg",
    "rec_yds_league_zscore", "rec_td_league_zscore", "rec_yds_league_rank",
]

DEFENSE_KICK_FEATURES = [
    # Defensive — league-ranked so a sack/INT leader stands out within a season
    "def_sacks", "def_int", "def_impact", "def_impact_pg",
    "def_sacks_league_zscore", "def_sacks_league_rank",
    "def_int_league_zscore", "def_int_league_rank",
    "def_impact_league_zscore", "def_impact_league_rank",
    "def_impact_pg_league_zscore",
    # Kicking
    "fg_made", "fg_pct", "kick_points",
    "fg_made_league_zscore", "fg_made_league_rank",
    "kick_points_league_zscore", "kick_points_league_rank",
]

CONTEXT_FEATURES = [
    "games", "full_season",
    "is_qb", "is_rb", "is_wr", "is_te", "is_def", "is_k",
    "pos_prior",
    "total_td", "scrimmage_yds", "total_td_pg", "scrimmage_yds_pg",
    "total_td_league_zscore", "total_td_league_rank",
    "scrimmage_yds_league_zscore", "scrimmage_yds_league_rank",
    "fantasy_pts_adj", "fantasy_pts_adj_pg",
    "fantasy_pts_adj_league_zscore", "fantasy_pts_adj_league_pct",
    "fantasy_pts_adj_league_rank", "fantasy_pts_adj_vs_league_avg",
    "dominance_score", "historically_dominant",
    "peak_dominance_zscore", "yardage_dominant", "max_yardage_zscore",
    "wins", "win_pct", "team_win_zscore", "team_win_pct_rank",
    "elite_team", "playoff_team", "win_pct_vs_avg",
    "late_fantasy_pg", "late_vs_early_fantasy", "late_season_trend",
    "strong_finish", "late_pass_epa_pg", "late_rush_yds_pg",
]

# ---------------------------------------------------------------------------
# Track feature sets
# ---------------------------------------------------------------------------
# The MVP race spans contests the single award covers. We model it as tracks:
#   qb      : quarterbacks (team-success gated)
#   skill   : RB / WR / TE (rewarded for carrying a heavy load)
#   offense : qb + skill merged (used by the legacy era, which does better merged)
#   defkick : defensive players + kickers (own isolated evaluation)
# A model is trained per track; scores are combined with dampened historical
# base rates so cross-track comparison (e.g. a QB vs an RB) is fair.

SHARED_CONTEXT = [
    "games", "full_season",
    "wins", "win_pct", "team_win_zscore", "team_win_pct_rank",
    "elite_team", "playoff_team", "win_pct_vs_avg",
    "pos_prior",
]

_SHARED_OFFENSE = [
    "total_td", "scrimmage_yds", "total_td_pg", "scrimmage_yds_pg",
    "total_td_league_zscore", "total_td_league_rank",
    "scrimmage_yds_league_zscore", "scrimmage_yds_league_rank",
    "fantasy_pts_adj", "fantasy_pts_adj_pg",
    "fantasy_pts_adj_league_zscore", "fantasy_pts_adj_league_pct",
    "fantasy_pts_adj_league_rank", "fantasy_pts_adj_vs_league_avg",
    "dominance_score", "historically_dominant",
    "peak_dominance_zscore", "yardage_dominant", "max_yardage_zscore",
    "late_fantasy_pg", "late_vs_early_fantasy", "late_season_trend",
    "strong_finish",
    # Supporting cast — carrying vs being carried
    "offense_share", "td_share", "carry_index",
    "team_total_yds", "team_total_td",
]

# League-leader flags. Ablation showed that leading the league in passing yards
# is a strong, non-redundant MVP signal (+8pts modern top-1); the other flags
# were noise or collinear with existing rank features, so we keep only the ones
# that earned their place.
QB_LEADER_FLAGS = ["led_pass_yds"]
SKILL_LEADER_FLAGS = ["led_rush_yds", "led_scrimmage"]

# QB track: passing-led, heavy team-success emphasis, QB rushing
QB_TRACK_FEATURES = (PASSING_FEATURES + [
    "rush_yds", "rush_td", "rush_yds_pg",
    "rush_yds_qb_pos_zscore", "rush_td_qb_pos_zscore", "rush_yds_qb_pos_pct",
    "qb_offense_share", "weapon_support", "best_weapon_yds", "supporting_yds",
    "late_pass_epa_pg",
] + QB_LEADER_FLAGS + _SHARED_OFFENSE + SHARED_CONTEXT)

# Skill track (RB/WR/TE): rushing/receiving-led, carry emphasis, lighter team gate
SKILL_TRACK_FEATURES = (RUSHING_FEATURES + RECEIVING_FEATURES + [
    "is_rb", "is_wr", "is_te",
    "skill_offense_share", "supporting_yds",
    "late_rush_yds_pg",
] + SKILL_LEADER_FLAGS + _SHARED_OFFENSE + SHARED_CONTEXT)

DEFKICK_FEATURES = DEFENSE_KICK_FEATURES + ["is_def", "is_k"] + SHARED_CONTEXT


def assign_track(position) -> str:
    """
    Map a position to its MVP contest track. QBs and skill players are split
    because the MVP formula differs by position: QBs are gated hard by team
    success (every modern QB MVP was on an elite team), while RBs/receivers
    can win by carrying a heavier individual load on a weaker team.

      qb      : quarterbacks
      skill   : RB / WR / TE
      defkick : defensive players + kickers
      none    : offensive line, punters, snappers — never MVP candidates
    """
    if pd.isna(position):
        return "none"
    p = str(position).upper().strip()
    if p == "QB" or p.startswith("QB"):
        return "qb"
    if p in {"RB", "HB", "FB", "WR", "TE"} or any(
            p.startswith(x) for x in ["RB", "HB", "FB", "WR", "TE"]):
        return "skill"
    if p in {"K", "PK"}:
        return "defkick"
    if p in {"DE", "DT", "LB", "CB", "S", "FS", "SS", "NT", "DB", "DL",
             "OLB", "ILB", "MLB", "EDGE", "SAF"}:
        return "defkick"
    return "none"


# Merged offense track (qb + skill together) — used by the legacy era
OFFENSE_FEATURES = list(dict.fromkeys(
    QB_TRACK_FEATURES + SKILL_TRACK_FEATURES + ["is_qb", "is_rb", "is_wr", "is_te"]
))

TRACK_FEATURES = {
    "qb": QB_TRACK_FEATURES,
    "skill": SKILL_TRACK_FEATURES,
    "offense": OFFENSE_FEATURES,
    "defkick": DEFKICK_FEATURES,
}

# Advanced stats that DON'T exist before ~1999. The legacy (pre-1999) model
# excludes these and relies on traditional box-score stats + team record.
ADVANCED_ONLY = {
    "pass_epa", "pass_epa_pg", "pass_epa_league_zscore", "pass_epa_league_rank",
    "pass_epa_vs_league_avg", "pass_epa_qb_pos_zscore", "late_pass_epa_pg",
    "pacr", "dakota", "rush_epa", "rec_epa",
    "target_share", "wopr", "racr",
}


# League-leader flags help the modern split model but add noise to the legacy
# merged model, so they're excluded from the legacy feature set.
ALL_LEADER_FLAGS = set(QB_LEADER_FLAGS) | set(SKILL_LEADER_FLAGS)


def get_feature_cols(df: pd.DataFrame, track: str, era: str = "modern") -> list:
    """
    Return feature columns for a given track + era that exist in df.
      track : qb / skill / offense / defkick
      era   : "modern" (incl. EPA/advanced + leader flags) or
              "legacy" (traditional only, no advanced stats or leader flags)
    """
    base = TRACK_FEATURES[track]
    if era == "legacy":
        base = [f for f in base if f not in ADVANCED_ONLY and f not in ALL_LEADER_FLAGS]
    seen, cols = set(), []
    for c in base:
        if c in df.columns and c not in seen:
            seen.add(c)
            cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _track_frame(df: pd.DataFrame, track: str, era: str, strategy: str):
    """Subset df to one track (per the era's strategy) and return (frame, feature_cols)."""
    sub = df[df["position"].apply(lambda p: assign_track_for(p, strategy)) == track].copy()
    feature_cols = get_feature_cols(sub, track, era)
    return sub.reset_index(drop=True), feature_cols


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.05,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Era definitions: name -> (start, end, era-flag, track-strategy)
# track-strategy controls how positions are grouped into models:
#   "split"  : separate qb / skill / defkick models (modern benefits from this —
#              QBs are team-gated, skill players reward carrying)
#   "merged" : qb+skill share one "offense" model, defkick separate (legacy does
#              better merged: its smaller per-era data can't support the split)
ERAS = {
    "modern": (1999, 2024, "modern", "split"),
    "legacy": (1957, 1998, "legacy", "merged"),
}


def tracks_for_strategy(strategy: str):
    """Return the list of track names for a given strategy."""
    if strategy == "split":
        return ["qb", "skill", "defkick"]
    return ["offense", "defkick"]


def assign_track_for(position, strategy: str) -> str:
    """Assign a position to a track given the era's strategy."""
    t = assign_track(position)          # qb / skill / defkick / none
    if strategy == "merged" and t in ("qb", "skill"):
        return "offense"
    return t


# Dampening exponent for cross-track base rates. Offense wins ~97% of MVPs;
# a raw multiplier would make a defender essentially unrankable. We dampen so a
# genuinely historic defensive/kicking season (LT '86, Moseley '82) can surface.
BASE_RATE_POWER = 0.35


# Training target mode:
#   "binary"     : positive = MVP winner only (original)
#   "vote_share" : treat every vote-getter as a fractional positive, weighted by
#                  their vote share. Uses the full exported voting data as a
#                  graded target — the runner-up counts, close races are learned.
# NOTE: "vote_share" mode produced good-looking CV ranking numbers but broke the
# absolute predictions — switching the target to "got any votes" compressed all
# probabilities near 0.5 and let obscure skill players outscore the real winner.
# Binary (winner-only) gives correctly separated, trustworthy predictions, so it
# is the default. The voting data is still loaded and available for future use.
TARGET_MODE = "binary"

# Minimum weight floor for the actual winner so they're never under-weighted.
WINNER_MIN_WEIGHT = 1.0

# How strongly vote share scales a candidate's training weight. Tuned by CV:
# 2 gives the best top-1/top-3/top-5 together.
VOTE_SHARE_MULT = 2.0


def _fit_track(df, track, era, strategy, seasons_in_era):
    """Fit one track's pipeline; returns (pipeline, feature_cols, base_rate)."""
    sub, feats = _track_frame(df, track, era, strategy)
    if sub.empty or sub["mvp_winner"].sum() == 0:
        return None, feats, 0.0
    X = sub[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

    if TARGET_MODE == "vote_share" and "vote_share" in sub.columns and sub["vote_share"].sum() > 0:
        # Graded target: label every vote-getter as positive, weighted by their
        # vote share. A near-unanimous winner contributes ~1.0; a player who got
        # 15% of the vote contributes 0.15 toward the positive class. Non-vote
        # getters are negatives at weight 1.0.
        y = sub["got_votes"].astype(int) if "got_votes" in sub.columns else sub["mvp_winner"].astype(int)
        vs = pd.to_numeric(sub["vote_share"], errors="coerce").fillna(0.0).values
        is_winner = (sub["mvp_winner"] == 1).values
        sw = np.ones(len(sub), dtype=float)
        pos = y.values == 1
        # positives weighted by vote share (winner floored so it's never tiny).
        # Multiplier of 2 tuned by CV: best top-1/top-3/top-5 simultaneously.
        sw[pos] = np.maximum(vs[pos] * VOTE_SHARE_MULT, 0.25)
        sw[is_winner] = np.maximum(sw[is_winner], WINNER_MIN_WEIGHT)
        pipe = build_logistic_pipeline()
        pipe.fit(X, y, clf__sample_weight=sw)
    else:
        y = sub["mvp_winner"].astype(int)
        pipe = build_logistic_pipeline()
        pipe.fit(X, y)

    raw_rate = sub["mvp_winner"].sum() / seasons_in_era
    base_rate = float(raw_rate ** BASE_RATE_POWER)
    return pipe, feats, base_rate


def train(era: str = "modern", df: pd.DataFrame = None) -> dict:
    """
    Train a two-track MVP model (offense + defense/kick) for one era,
    cross-validate by leave-one-season-out, and save.
    """
    if era not in ERAS:
        raise ValueError(f"Unknown era '{era}'. Choose from {list(ERAS)}.")
    start_season, end_season, era_flag, strategy = ERAS[era]
    tracks = tracks_for_strategy(strategy)

    if df is None:
        features_path = PROCESSED_DIR / "features.csv"
        if not features_path.exists():
            raise FileNotFoundError(
                "features.csv not found — run `python run.py features` first."
            )
        df = pd.read_csv(features_path, low_memory=False)

    df = df[(df["season"] >= start_season) & (df["season"] <= end_season)].copy()
    df = df.reset_index(drop=True)
    df["_track"] = df["position"].apply(lambda p: assign_track_for(p, strategy))
    counts = {t: int(df[df["_track"] == t]["mvp_winner"].sum()) for t in tracks}
    print(f"[{era}] {len(df)} player-seasons ({start_season}–{end_season}), "
          f"strategy={strategy}; MVPs by track: {counts}")

    seasons = sorted(df["season"].unique())

    # ---- Leave-one-season-out CV across all tracks ----
    cv_score = np.zeros(len(df))
    for s in seasons:
        train_df = df[df["season"] != s]
        test_df = df[df["season"] == s]
        n_train_seasons = train_df["season"].nunique()
        for track in tracks:
            pipe, feats, base = _fit_track(train_df, track, era_flag, strategy, n_train_seasons)
            if pipe is None:
                continue
            te = test_df[test_df["_track"] == track]
            if te.empty:
                continue
            X = te[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
            p = pipe.predict_proba(X)[:, 1] * base
            cv_score[te.index.values] = p

    df["cv_score"] = cv_score

    # Metrics (ranking within each full season across both tracks)
    top_n_acc = {}
    for n in [1, 2, 3, 5]:
        correct = total = 0
        for s, grp in df.groupby("season"):
            if grp["mvp_winner"].sum() == 0:
                continue
            top = grp.nlargest(n, "cv_score")
            if top["mvp_winner"].max() == 1:
                correct += 1
            total += 1
        top_n_acc[f"top_{n}_acc"] = round(correct / total, 3) if total else 0

    try:
        auc = roc_auc_score(df["mvp_winner"], df["cv_score"])
    except Exception:
        auc = float("nan")

    print(f"\n=== CV Results ({era}: {start_season}–{end_season}) ===")
    print(f"  ROC-AUC:      {auc:.4f}")
    for k, v in top_n_acc.items():
        print(f"  {k}:      {v:.1%}")

    # ---- Fit final per-track models on full era ----
    print("\nFitting final models on full era dataset...")
    n_all_seasons = df["season"].nunique()
    track_models = {}
    for track in tracks:
        pipe, feats, base = _fit_track(df, track, era_flag, strategy, n_all_seasons)
        if pipe is None:
            continue
        sub, _ = _track_frame(df, track, era_flag, strategy)
        Xc = sub[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        yc = sub["mvp_winner"].astype(int)
        # Calibrate only when there are enough positive examples for CV folds;
        # otherwise use the raw pipeline (the skill/defkick tracks can have <5 MVPs).
        n_pos = int(yc.sum())
        if n_pos >= 8:
            final = CalibratedClassifierCV(pipe, cv=min(5, n_pos), method="isotonic")
            final.fit(Xc, yc)
        else:
            final = pipe  # already fitted
        track_models[track] = {
            "model": final, "feature_cols": feats, "base_rate": base,
        }

    bundle = {"era": era, "tracks": track_models, "base_rate_power": BASE_RATE_POWER}
    model_path = MODEL_DIR / f"mvp_model_{era}.pkl"
    meta_path = MODEL_DIR / f"mvp_model_{era}_meta.json"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    meta = {
        "era": era, "model_type": "logistic-two-track",
        "start_season": start_season, "end_season": end_season,
        "tracks": {t: {"base_rate": round(v["base_rate"], 4),
                       "n_features": len(v["feature_cols"])}
                   for t, v in track_models.items()},
        "roc_auc": round(auc, 4), **top_n_acc,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {model_path}")
    return {"model": bundle, "metrics": meta}


def train_all():
    """Train both the modern and legacy models."""
    results = {}
    for era in ["modern", "legacy"]:
        print(f"\n{'='*60}\nTraining {era} model\n{'='*60}")
        results[era] = train(era=era)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", choices=["modern", "legacy", "all"], default="modern")
    args = parser.parse_args()
    if args.era == "all":
        train_all()
    else:
        train(era=args.era)
