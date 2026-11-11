"""
voter_bias.py
-------------
Features that capture SYSTEMATIC voter tendencies that never show up in a box
score. Voters are not random: they reward reputation, get tired of repeat
winners, love comeback and turnaround stories. These are learnable if we give
the model the history to see them.

Every feature uses only information available BEFORE the season being scored
(prior-year votes, past wins, last year's games played, last year's team
record), so there is no leakage from the answer.

History comes from the stored data files, so the same function works for
training rows and for a live prediction pool.

STATUS: tested, NOT used by the pipeline. With 27 seasons (28 winners) the
features gave no measurable gain: 8-seed paired CV showed changes in mean
reciprocal rank of -0.007 to +0.008 (all within one standard error), and a
residual analysis (does a feature explain the vote share the model misses?)
gave 95% intervals spanning zero for all ten. The reputation features
(prior_vote_share, is_defending_mvp, prior_mvp_wins) lean positive, at roughly
+2 points of vote share per SD, but are indistinguishable from noise at this
sample size.

A second, fairer attempt used a two-stage design (the ranker's score plus a
few bias features in a small ridge re-ranker over each year's top 8, with fully
nested leave-one-season-out). It scored worse than the ranker alone for every
feature set (change in MRR -0.015 to -0.037, bootstrap intervals over seasons
spanning zero, seasons improved vs worsened roughly even), and the learned
weights on correlated features were unstable. Worth re-testing as more seasons
accumulate.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

BIAS_FEATURES = [
    # Reputation / star power
    "prior_vote_share",       # share of the vote the player got last season
    "prior3_vote_share_sum",  # total vote share over the last 3 seasons
    "prior_vote_seasons",     # seasons (since 1999) with any MVP votes
    # Voter fatigue / "overdue"
    "prior_mvp_wins",         # MVPs won before this season
    "years_since_win",        # seasons since last MVP (30 = never)
    "is_defending_mvp",       # won last season
    # Comeback story
    "prev_games_missed",      # games missed last season
    "comeback",               # missed 6+ games last season
    "no_prior_season",        # no stats last season (rookie / new)
    # Turnaround story
    "win_pct_change",         # team win% minus the same team's win% last year
]


def _clean(name) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[*+]", "", str(name)).strip()


def _regular_season_games(season: int) -> int:
    if season <= 1960:
        return 12
    if season <= 1977:
        return 14
    if season <= 2020:
        return 16
    return 17


def _load_history() -> dict:
    """Load the lookup tables the bias features are built from."""
    hist = {}

    # MVP wins by player -> sorted array of seasons won
    mvp = pd.read_csv(RAW_DIR / "mvp_winners.csv")
    mvp["player"] = mvp["player"].apply(_clean)
    hist["wins"] = {p: np.sort(g["season"].astype(int).values)
                    for p, g in mvp.groupby("player")}

    # Vote share by (player, season), and seasons-with-votes by player
    try:
        from votes import load_all_votes
        votes = load_all_votes()
    except Exception:
        votes = pd.DataFrame(columns=["season", "player", "vote_share"])
    if not votes.empty:
        votes["player"] = votes["player"].apply(_clean)
        votes["vote_share"] = pd.to_numeric(votes["vote_share"], errors="coerce").fillna(0.0)
        vs = votes.groupby(["player", "season"])["vote_share"].sum()
        hist["vote_share"] = vs.to_dict()
        hist["vote_seasons"] = {p: np.sort(g["season"].astype(int).values)
                                for p, g in votes[votes["vote_share"] > 0].groupby("player")}
    else:
        hist["vote_share"], hist["vote_seasons"] = {}, {}

    # Games played by (player, season)
    stats = pd.read_csv(RAW_DIR / "player_stats.csv", usecols=["season", "player", "games_played"])
    stats["player"] = stats["player"].apply(_clean)
    hist["games"] = stats.groupby(["player", "season"])["games_played"].max().to_dict()

    # Team win% by (team, season)
    teams = pd.read_csv(RAW_DIR / "team_records.csv", usecols=["season", "team", "win_pct"])
    hist["team_win_pct"] = teams.set_index(["team", "season"])["win_pct"].to_dict()
    return hist


def add_voter_bias_features(df: pd.DataFrame, hist: dict = None) -> pd.DataFrame:
    """Append the BIAS_FEATURES columns to df (needs player, season, team, win_pct)."""
    hist = hist or _load_history()
    out = df.copy()
    players = out["player"].apply(_clean).values
    seasons = out["season"].astype(int).values
    teams = out["team"].values if "team" in out.columns else np.full(len(out), None)

    vshare, vseasons, wins = hist["vote_share"], hist["vote_seasons"], hist["wins"]
    games, twp = hist["games"], hist["team_win_pct"]

    prior_vs, prior3, prior_nseas = [], [], []
    n_wins, since_win, defending = [], [], []
    missed, comeback, no_prior, wpc = [], [], [], []

    cur_wp = out["win_pct"].values if "win_pct" in out.columns else np.full(len(out), np.nan)

    for p, s, t, wp in zip(players, seasons, teams, cur_wp):
        prior_vs.append(vshare.get((p, s - 1), 0.0))
        prior3.append(sum(vshare.get((p, s - k), 0.0) for k in (1, 2, 3)))
        arr = vseasons.get(p)
        prior_nseas.append(int((arr < s).sum()) if arr is not None else 0)

        w = wins.get(p)
        prior = w[w < s] if w is not None else np.array([], dtype=int)
        n_wins.append(len(prior))
        since_win.append(int(s - prior.max()) if len(prior) else 30)
        defending.append(int((s - 1) in prior))

        g_prev = games.get((p, s - 1))
        if g_prev is None:
            no_prior.append(1)
            missed.append(0)
            comeback.append(0)
        else:
            m = max(0, _regular_season_games(s - 1) - int(g_prev))
            no_prior.append(0)
            missed.append(m)
            comeback.append(int(m >= 6))

        prev_wp = twp.get((t, s - 1))
        wpc.append(float(wp - prev_wp) if (prev_wp is not None and pd.notna(wp)) else 0.0)

    out["prior_vote_share"] = prior_vs
    out["prior3_vote_share_sum"] = prior3
    out["prior_vote_seasons"] = prior_nseas
    out["prior_mvp_wins"] = n_wins
    out["years_since_win"] = since_win
    out["is_defending_mvp"] = defending
    out["prev_games_missed"] = missed
    out["comeback"] = comeback
    out["no_prior_season"] = no_prior
    out["win_pct_change"] = wpc
    return out
