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


def load_model():
    model_path = PROCESSED_DIR / "mvp_model_modern.pkl"
    meta_path = PROCESSED_DIR / "mvp_model_modern_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python run.py train` first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def predict_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a pool of players with the trained ranker and return the DataFrame
    with an `mvp_prob` column (readable odds), sorted best-first.

    Only offensive players (QB/RB/WR/TE) are scored — the modern model doesn't
    cover defenders/kickers (no non-offensive MVP since 1986).
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent))
    import features_spec as spec

    bundle, meta = load_model()
    tracks = bundle["tracks"]            # {"offense": {...}}

    df = df.copy()
    df["_track"] = (df["position"].apply(spec.assign_track)
                    if "position" in df.columns else "qb")
    # Route qb + skill into the single "offense" ranker; others are out of scope
    df["_route"] = df["_track"].apply(lambda t: "offense" if t in ("qb", "skill") else t)

    raw = np.full(len(df), -np.inf)
    for track, info in tracks.items():
        mask = (df["_route"] == track).values
        if not mask.any():
            continue
        feats = info["feature_cols"]
        for col in feats:
            if col not in df.columns:
                df[col] = 0.0
        X = df.loc[mask, feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        scores = info["ranker"].predict(info["scaler"].transform(X.values))
        raw[np.where(mask)[0]] = scores * info.get("base_rate", 1.0)

    df["mvp_score"] = raw
    # Ranking scores are arbitrary-scale; the ORDER is what matters. Convert to
    # readable odds with a softmax over the realistic contender pool (top 15)
    # so the probability mass isn't spread across hundreds of players.
    finite = np.isfinite(raw)
    work = np.where(finite, raw, (np.nanmin(raw[finite]) - 1) if finite.any() else 0)
    order = np.argsort(-work)
    pool = order[:15]
    probs = np.zeros(len(df))
    ps = work[pool]
    z = (ps - ps.mean()) / (ps.std() + 1e-9)
    exp = np.exp(z * 1.3)
    probs[pool] = exp / exp.sum()
    df["mvp_prob"] = probs
    df["mvp_prob_raw"] = raw

    result = df.sort_values("mvp_prob", ascending=False).reset_index(drop=True)
    result.index += 1
    return result


def format_output(result: pd.DataFrame) -> str:
    """Pretty-print the prediction results."""
    lines = ["\n=== NFL MVP Probability Rankings ===\n"]
    for i, row in result.iterrows():
        player = row.get("player", f"Player {i}")
        team = row.get("team", "?")
        pos = row.get("position", "?")
        prob = row.get("mvp_prob", 0)
        lines.append(f"  {i:2}. {player:<25} {pos:<5} {team:<5}  {prob:6.1%}")
    lines.append("\nOdds are a softmax over the top contenders — ordering is the "
                 "meaningful signal. Offensive players only (QB/RB/WR/TE).")
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
