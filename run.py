"""
run.py — command-line interface for the NFL MVP prediction project (v2).

Data flow:
  ingest   → process the local game-by-game parquet into raw season CSVs
  mvp      → scrape AP MVP winners from Wikipedia
  features → build the engineered feature dataset
  train    → train the learning-to-rank MVP model (1999–2025)
  predict  → predict MVP odds for a season or a custom set of players

Common commands:
  python run.py pipeline                              # ingest → mvp → features → train
  python run.py predict --season 2026                 # odds for a stored season
  python run.py predict --input players.csv           # odds for a custom player pool
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_ingest(args):
    from ingest_local import ingest
    ingest()


def cmd_mvp(args):
    from scraper import scrape_mvp_winners
    scrape_mvp_winners()


def cmd_features(args):
    from features import build_features
    build_features()


def cmd_train(args):
    import model_ranker
    model_ranker.train()


def cmd_predict(args):
    import pandas as pd
    from predict import (predict_from_df, add_derived_features,
                         format_output, collect_player_interactive)

    if args.season is not None:
        df = _season_pool(args.season)
        print(f"Loaded {len(df)} players for {args.season} from stored data.")
    elif args.input:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} players from {args.input}")
    else:
        players = []
        while True:
            players.append(collect_player_interactive())
            again = input("\nAdd another player? [y/N]: ").strip().lower()
            if again != "y":
                break
        df = pd.DataFrame(players)

    df = add_derived_features(df, use_weekly=False)
    result = predict_from_df(df)
    print(format_output(result.head(args.top)))

    if args.output:
        result.to_csv(args.output, index=True)
        print(f"\nResults saved to {args.output}")


def _season_pool(season: int):
    """Assemble a prediction pool from stored raw stats + team records."""
    import pandas as pd
    raw_dir = Path(__file__).parent / "data" / "raw"
    stats = pd.read_csv(raw_dir / "player_stats.csv")
    pool = stats[stats["season"] == season].copy()
    if pool.empty:
        raise SystemExit(f"No stored stats for season {season}.")
    if "games_played" in pool.columns:
        pool = pool.rename(columns={"games_played": "games"})
    teams = pd.read_csv(raw_dir / "team_records.csv")
    tseason = teams[teams["season"] == season][["team", "wins", "win_pct"]]
    pool = pool.merge(tseason, on="team", how="left")
    return pool


def cmd_pipeline(args):
    from ingest_local import ingest
    from scraper import scrape_mvp_winners
    from features import build_features
    import model_ranker

    print("Step 1/4: Ingesting local game data...")
    ingest()
    print("\nStep 2/4: Scraping MVP winners...")
    scrape_mvp_winners()
    print("\nStep 3/4: Building features...")
    build_features()
    print("\nStep 4/4: Training the ranker...")
    model_ranker.train()
    print("\n\u2713 Pipeline complete. Run `python run.py predict --season 2026` to predict.")


def main():
    parser = argparse.ArgumentParser(
        description="NFL MVP Prediction (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ingest", help="Process the local game-by-game parquet into raw CSVs")
    sub.add_parser("mvp", help="Scrape AP MVP winners from Wikipedia")
    sub.add_parser("features", help="Build the feature dataset")
    sub.add_parser("train", help="Train the learning-to-rank MVP model")

    p_pred = sub.add_parser("predict", help="Predict MVP probabilities")
    p_pred.add_argument("--season", type=int, default=None,
                        help="Predict for a stored season (builds the pool automatically)")
    p_pred.add_argument("--input", type=str, default=None, help="CSV with player stats")
    p_pred.add_argument("--output", type=str, default=None, help="Save results to CSV")
    p_pred.add_argument("--top", type=int, default=10, help="How many players to show")

    sub.add_parser("pipeline", help="Run ingest → mvp → features → train")

    args = parser.parse_args()
    dispatch = {
        "ingest": cmd_ingest, "mvp": cmd_mvp, "features": cmd_features,
        "train": cmd_train, "predict": cmd_predict, "pipeline": cmd_pipeline,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
