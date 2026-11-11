"""
schedule.py
-----------
Schedule-based features built from the nflverse games file (1999+):

  Strength of schedule
    sos_opp_win_pct       average win% of the opponents a team has faced
    opp_def_weakness      avg points allowed by those opponents minus the league
                          average (positive = faced weak defenses)
    quality_wins          wins against opponents with a .600+ record
    sched_adj_fantasy_pg  the player's fantasy points per game, scaled up when the
                          defenses faced were tough and down when they were weak

  Primetime exposure
    primetime_games       games the team played at a 7pm ET or later kickoff
    primetime_fp_delta    player's fantasy points per game in primetime games
                          minus his average over all games (0 if none)

STATUS: tested, not used by the model. 8-seed paired ablation vs baseline:
sos dMRR +0.002, primetime +0.005, all six +0.001 (each within 1 SE of zero,
SE about 0.011). In the residual check, 5 of 6 features had CIs spanning zero;
sched_adj_fantasy_pg showed +5.7 pts [+1.3, +9.8] but it was 1 hit out of 6
tests and adding it did not improve ranking, so it is treated as unconfirmed.

Everything is computed from games already played, so it works for a season in
progress. 1999 has no kickoff times in the source, so primetime features are 0
for that season.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
GAMES_PATH = RAW_DIR / "nflverse_games.parquet"
GAMES_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "schedules/games.parquet")

TEAM_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}   # match player-stats team codes
PRIMETIME_HOUR = 19
QUALITY_OPP_WIN_PCT = 0.600

SCHEDULE_FEATURES = [
    "sos_opp_win_pct", "opp_def_weakness", "quality_wins", "sched_adj_fantasy_pg",
    "primetime_games", "primetime_fp_delta",
]


def _clean(name) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[*+]", "", str(name)).strip()


def download_games() -> pd.DataFrame:
    import io
    import requests
    import pyarrow.parquet as pq
    resp = requests.get(GAMES_URL, headers={"User-Agent": "research python-requests"},
                        timeout=120)
    resp.raise_for_status()
    g = pq.read_table(io.BytesIO(resp.content)).to_pandas()
    g.to_parquet(GAMES_PATH, index=False)
    return g


def load_games(refresh: bool = False) -> pd.DataFrame:
    if refresh or not GAMES_PATH.exists():
        return download_games()
    return pd.read_parquet(GAMES_PATH)


def team_games(games: pd.DataFrame = None) -> pd.DataFrame:
    """One row per team per regular-season game (both sides of every game)."""
    g = load_games() if games is None else games
    g = g[g["game_type"] == "REG"].copy()
    hour = pd.to_numeric(g["gametime"].astype(str).str.slice(0, 2), errors="coerce")
    g["prime"] = (hour >= PRIMETIME_HOUR).astype(int)
    g["played"] = g["home_score"].notna() & g["away_score"].notna()
    g["home_team"] = g["home_team"].replace(TEAM_MAP)
    g["away_team"] = g["away_team"].replace(TEAM_MAP)

    home = pd.DataFrame({"season": g.season, "week": g.week, "team": g.home_team,
                         "opp": g.away_team, "pf": g.home_score, "pa": g.away_score,
                         "prime": g.prime, "played": g.played})
    away = pd.DataFrame({"season": g.season, "week": g.week, "team": g.away_team,
                         "opp": g.home_team, "pf": g.away_score, "pa": g.home_score,
                         "prime": g.prime, "played": g.played})
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg[tg["played"]].copy()
    tg["win"] = np.where(tg.pf > tg.pa, 1.0, np.where(tg.pf == tg.pa, 0.5, 0.0))
    return tg


def team_schedule_table(games: pd.DataFrame = None) -> pd.DataFrame:
    """Per (season, team) schedule features, using games played to date."""
    tg = team_games(games)

    # Each team's record and defense to date
    to_date = tg.groupby(["season", "team"]).agg(
        gp=("win", "size"), wins=("win", "sum"), pa_pg=("pa", "mean")).reset_index()
    to_date["win_pct"] = to_date.wins / to_date.gp
    league_pa = tg.groupby("season")["pa"].mean().rename("league_pa").reset_index()

    opp = to_date.rename(columns={"team": "opp", "win_pct": "opp_win_pct",
                                  "pa_pg": "opp_pa_pg"})[["season", "opp", "opp_win_pct", "opp_pa_pg"]]
    tg = tg.merge(opp, on=["season", "opp"], how="left")
    tg["quality_win"] = ((tg.win == 1.0) & (tg.opp_win_pct >= QUALITY_OPP_WIN_PCT)).astype(int)

    out = tg.groupby(["season", "team"]).agg(
        sos_opp_win_pct=("opp_win_pct", "mean"),
        opp_pa_pg_mean=("opp_pa_pg", "mean"),
        quality_wins=("quality_win", "sum"),
        primetime_games=("prime", "sum"),
    ).reset_index().merge(league_pa, on="season", how="left")
    out["opp_def_weakness"] = out.opp_pa_pg_mean - out.league_pa
    # >1 when the defenses faced allowed fewer points than average (tough schedule)
    out["def_difficulty"] = out.league_pa / out.opp_pa_pg_mean
    return out[["season", "team", "sos_opp_win_pct", "opp_def_weakness",
                "quality_wins", "primetime_games", "def_difficulty"]]


def player_primetime_delta(keys: pd.DataFrame, games: pd.DataFrame = None) -> pd.Series:
    """
    For each (player, season) in `keys`, the player's position-adjusted fantasy
    points per game in primetime games minus his average over all games.
    0 when he played no primetime games. Needs the weekly stats file.
    """
    weekly_path = RAW_DIR / "player_stats_weekly.csv"
    if not weekly_path.exists():
        return pd.Series(0.0, index=keys.index)

    cols = ["season", "week", "player", "team", "pass_yds", "pass_td", "pass_int",
            "rush_yds", "rush_td", "rec", "rec_yds", "rec_td"]
    w = pd.read_csv(weekly_path, usecols=cols)
    w = w[w["season"] >= 1999].copy()
    for c in cols[4:]:
        w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)
    w["fp"] = (w.pass_yds * 0.04 + w.rush_yds * 0.1 + w.rec_yds * 0.1 +
               6 * (w.pass_td + w.rush_td + w.rec_td) + 0.5 * w.rec - 2 * w.pass_int)
    w["player"] = w["player"].apply(_clean)

    tg = team_games(games)[["season", "week", "team", "prime"]].drop_duplicates()
    w = w.merge(tg, on=["season", "week", "team"], how="left")
    w["prime"] = w["prime"].fillna(0)

    overall = w.groupby(["player", "season"])["fp"].mean().rename("fp_all")
    prime = (w[w.prime == 1].groupby(["player", "season"])["fp"].mean().rename("fp_prime"))
    t = pd.concat([overall, prime], axis=1)
    t["delta"] = (t.fp_prime - t.fp_all).fillna(0.0)

    idx = pd.MultiIndex.from_arrays([keys["player"].apply(_clean).values,
                                     keys["season"].astype(int).values])
    return pd.Series(t["delta"].reindex(idx).fillna(0.0).values, index=keys.index)


def add_schedule_features(df: pd.DataFrame, games: pd.DataFrame = None) -> pd.DataFrame:
    """Append SCHEDULE_FEATURES to a player-season frame (needs season, team, player)."""
    tbl = team_schedule_table(games)
    out = df.copy()
    out = out.merge(tbl, on=["season", "team"], how="left")

    fp_pg = out["fantasy_pts_adj_pg"] if "fantasy_pts_adj_pg" in out.columns else np.nan
    out["sched_adj_fantasy_pg"] = (fp_pg * out["def_difficulty"]).round(3)
    out["primetime_fp_delta"] = player_primetime_delta(out, games).round(3).values
    return out.drop(columns=["def_difficulty"])
