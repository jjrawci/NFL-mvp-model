"""
predict.py
----------
Given a set of player stats (from a CSV or typed in manually),
load the trained model and output MVP probability for each player.

Usage examples:
  # Predict from a CSV of current-season stats
  python src/predict.py --input data/current_season.csv

  # Interactive mode — type stats one player at a time
  python src/predict.py --interactive

  # Use a specific model
  python src/predict.py --input data/current_season.csv --model xgboost
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_model(era: str = "modern"):
    model_path = PROCESSED_DIR / f"mvp_model_{era}.pkl"
    meta_path = PROCESSED_DIR / f"mvp_model_{era}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run `python run.py train --era {era}` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta


def _assign_track(position) -> str:
    """Mirror of model.assign_track — map a position to its MVP contest track."""
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return "qb"
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
    return "qb"


def predict_from_df(df: pd.DataFrame, era: str = "modern") -> pd.DataFrame:
    """
    Given a DataFrame of player stats, score each player with the appropriate
    track model (offense or defense/kick), combine via base rates, and return
    the DataFrame with an `mvp_prob` column sorted by probability.

    era="modern" uses the 1999+ models, era="legacy" the pre-1999 models.
    """
    bundle, meta = load_model(era)
    tracks = bundle["tracks"]
    is_ranker = bundle.get("kind") == "ranker"
    merged = "offense" in tracks   # legacy merges qb+skill into one "offense" track

    # Both the legacy merged model and the modern ranker use one "offense" track
    has_offense = "offense" in tracks

    def route(pos):
        t = _assign_track(pos)     # qb / skill / defkick
        if has_offense and t in ("qb", "skill"):
            return "offense"
        return t

    df = df.copy()
    df["_track"] = df["position"].apply(route) if "position" in df.columns else (
        "offense" if merged else "qb")
    raw = np.zeros(len(df))

    for track, info in tracks.items():
        mask = (df["_track"] == track).values
        if not mask.any():
            continue
        feats = info["feature_cols"]
        for col in feats:
            if col not in df.columns:
                df[col] = 0.0
        X = df.loc[mask, feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        if is_ranker:
            # Ranker outputs an arbitrary-scale score; standardise within the
            # pool then squash to a positive comparable scale, times base rate.
            scaler = info["scaler"]
            scores = info["ranker"].predict(scaler.transform(X.values))
            raw[np.where(mask)[0]] = scores * info["base_rate"]
        else:
            p = info["model"].predict_proba(X)[:, 1] * info["base_rate"]
            raw[np.where(mask)[0]] = p

    # The modern ranker only covers offense; zero out anyone not scored by a
    # track (e.g. defenders) so they don't appear in the rankings.
    if is_ranker:
        scored = df["_track"].isin(tracks).values
        raw[~scored] = -np.inf

    df["mvp_score"] = raw
    if is_ranker:
        # Ranking scores are arbitrary-scale; the ORDER is what's meaningful.
        # Turn them into readable odds with a softmax, but only over the
        # realistic contender pool (top ~15 by score) so the probability mass
        # isn't diluted across hundreds of irrelevant players.
        raw_finite = np.where(np.isfinite(raw), raw, np.nanmin(raw[np.isfinite(raw)]) - 1)
        raw = raw_finite
        order = np.argsort(-raw)
        pool = order[:15]
        probs = np.zeros(len(df))
        pool_scores = raw[pool]
        z = (pool_scores - pool_scores.mean()) / (pool_scores.std() + 1e-9)
        exp = np.exp(z * 1.3)
        probs[pool] = exp / exp.sum()
        df["mvp_prob"] = probs
    else:
        total = raw.sum()
        df["mvp_prob"] = raw / total if total > 0 else raw
    df["mvp_prob_raw"] = raw

    result = df.sort_values("mvp_prob", ascending=False).reset_index(drop=True)
    result.index += 1
    return result


def format_output(result: pd.DataFrame) -> str:
    """Pretty-print the prediction results."""
    lines = ["\n=== NFL MVP Probability Rankings ===\n"]
    id_cols = [c for c in ["player", "team", "position", "year"] if c in result.columns]

    for i, row in result.iterrows():
        player = row.get("player", f"Player {i}")
        team = row.get("team", "?")
        pos = row.get("position", "?")
        prob = row.get("mvp_prob", 0)
        lines.append(f"  {i:2}. {player:<25} {pos:<5} {team:<5}  {prob:6.1%}")

    lines.append("\nNote: probabilities are normalised within this candidate pool.")
    lines.append(f"Model era: {result.attrs.get('era', 'modern')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive input mode
# ---------------------------------------------------------------------------

INTERACTIVE_FIELDS = {
    # (prompt, default, type)
    "player":      ("Player name", "", str),
    "position":    ("Position (QB/RB/WR/TE/DE/K/etc.)", "QB", str),
    "team":        ("Team abbreviation", "", str),
    "games":       ("Games played", 17, float),
    "wins":        ("Team wins", 9, float),
    "win_pct":     ("Team win %", 0.53, float),
    # QB stats
    "pass_yds":    ("Passing yards (0 if N/A)", 0, float),
    "pass_td":     ("Passing TDs", 0, float),
    "pass_int":    ("Interceptions", 0, float),
    "pass_rating": ("Passer rating", 0, float),
    "pass_cmp_pct":("Completion % (e.g. 0.67)", 0, float),
    "pass_epa":    ("Passing EPA (0 if unknown)", 0, float),
    # Rushing
    "rush_yds":    ("Rushing yards", 0, float),
    "rush_att":    ("Rushing attempts", 0, float),
    "rush_td":     ("Rushing TDs", 0, float),
    # Receiving
    "rec_yds":     ("Receiving yards", 0, float),
    "rec_td":      ("Receiving TDs", 0, float),
    "targets":     ("Targets", 0, float),
    "target_share":("Target share (0–1)", 0, float),
    # Fantasy
    "fantasy_pts": ("Fantasy points (standard)", 0, float),
}


def collect_player_interactive() -> dict:
    print("\n--- Enter player stats (press Enter to use default) ---")
    player_data = {}
    for field, (prompt, default, dtype) in INTERACTIVE_FIELDS.items():
        while True:
            raw = input(f"  {prompt} [{default}]: ").strip()
            if raw == "":
                player_data[field] = default
                break
            try:
                player_data[field] = dtype(raw)
                break
            except ValueError:
                print(f"    Invalid input, expected {dtype.__name__}")
    return player_data


def add_derived_features(df: pd.DataFrame, use_weekly: bool = False) -> pd.DataFrame:
    """
    Compute all model features for a prediction pool using the SAME pipeline as
    training (features.engineer_features), so prediction and training agree.

    The input should be the full candidate pool for the season — the relative
    features (league z-scores, ranks, carry/offense share) are computed across
    whoever is in the pool, so include all the realistic contenders.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from features import engineer_features

    df = df.copy()

    # Ensure required id columns exist
    if "season" not in df.columns:
        df["season"] = 2024
    if "team" not in df.columns:
        df["team"] = df.get("player", pd.Series(range(len(df)))).astype(str)
    if "games" not in df.columns:
        df["games"] = 17
    # mvp_winner is only used as a safeguard in build_features; add a dummy
    df["mvp_winner"] = 0

    return engineer_features(df, team_cast=None, use_weekly=use_weekly)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict NFL MVP probabilities")
    parser.add_argument("--input", type=str, help="Path to CSV with player stats")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter player stats interactively")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic",
                        help="Which trained model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to this CSV path")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} players from {args.input}")

    elif args.interactive:
        players = []
        while True:
            p = collect_player_interactive()
            players.append(p)
            again = input("\nAdd another player? [y/N]: ").strip().lower()
            if again != "y":
                break
        df = pd.DataFrame(players)

    else:
        parser.print_help()
        return

    df = add_derived_features(df)
    result = predict_from_df(df, model_type=args.model)
    result.attrs["model_type"] = args.model

    print(format_output(result))

    if args.output:
        result.to_csv(args.output, index=True)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
