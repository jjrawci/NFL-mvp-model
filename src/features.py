"""
features.py
-----------
Loads raw CSVs from data/raw/, cleans them, and builds the master
feature DataFrame used for training and prediction.

Each row = one player-season candidate.
Target column: mvp_winner (1/0).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Players must have played at least this fraction of the season median
# games to be included in season-level rankings/comparisons.
MIN_GAMES_FRACTION = 0.75

# Volume thresholds to define "comparable peers" per position.
# A player must exceed these to be counted as a full-volume starter.
# Used when computing position peer groups.
VOLUME_THRESHOLDS = {
    "QB":  {"pass_att": 400},          # starter-level QB
    "RB":  {"rush_att": 100},          # featured back
    "WR":  {"targets":   80},          # clear starter
    "TE":  {"targets":   40},          # featured TE
}

# Historical position prior — how often each position wins AP MVP.
# Deliberately small so a dominant non-QB season can still win.
POSITION_PRIOR = {
    "QB": 0.10,
    "RB": 0.04,
    "WR": 0.01,
    "TE": 0.01,
    "FB": 0.01,
    "LB": 0.01,
    "DT": 0.01,
    "DE": 0.01,
    "K":  0.01,
}
DEFAULT_PRIOR = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        print(f"  WARN: {filename} not found — skipping.")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _clean_name(name) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"[*+]", "", str(name)).strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_player_stats() -> pd.DataFrame:
    df = _safe_load("player_stats.csv")
    if df.empty:
        return df

    rename = {
        "player_display_name": "player",
        "season": "season",
        "recent_team": "team",
        "position": "position",
        "games_played": "games",
        "completions": "pass_cmp",
        "attempts": "pass_att",
        "passing_yards": "pass_yds",
        "passing_tds": "pass_td",
        "interceptions": "pass_int",
        "sacks": "pass_sacks",
        "passing_first_downs": "pass_first_downs",
        "passing_epa": "pass_epa",
        "pacr": "pacr",
        "dakota": "dakota",
        "carries": "rush_att",
        "rushing_yards": "rush_yds",
        "rushing_tds": "rush_td",
        "rushing_fumbles": "rush_fumbles",
        "rushing_first_downs": "rush_first_downs",
        "rushing_epa": "rush_epa",
        "receptions": "rec",
        "targets": "targets",
        "receiving_yards": "rec_yds",
        "receiving_tds": "rec_td",
        "receiving_fumbles": "rec_fumbles",
        "receiving_first_downs": "rec_first_downs",
        "receiving_epa": "rec_epa",
        "racr": "racr",
        "target_share": "target_share",
        "wopr": "wopr",
        "fantasy_points": "fantasy_pts",
        "fantasy_points_ppr": "fantasy_pts_ppr",
        "special_teams_tds": "special_teams_tds",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "player" not in df.columns and "player_id" in df.columns:
        df["player"] = df["player_id"]

    df["player"] = df["player"].apply(_clean_name)
    df = df[df["player"] != ""]

    non_numeric = ["player", "player_id", "team", "position", "position_group", "season"]
    for c in df.columns:
        if c not in non_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_team_records() -> pd.DataFrame:
    df = _safe_load("team_records.csv")
    if df.empty:
        return df
    for c in ["wins", "losses", "ties", "games", "win_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_mvp_winners() -> pd.DataFrame:
    df = _safe_load("mvp_winners.csv")
    if df.empty:
        return df
    df["player"] = df["player"].apply(_clean_name)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    df["mvp_winner"] = 1
    return df[["season", "player", "position", "mvp_winner"]].copy()


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features() -> pd.DataFrame:
    """
    Merge all data sources into one feature DataFrame.
    Each row is a player-season. MVP winners are labelled 1.
    """
    print("Building feature set...")

    mvp   = load_mvp_winners()
    stats = load_player_stats()
    teams = load_team_records()

    if stats.empty:
        raise RuntimeError("No player stats found — run `python run.py scrape` first.")

    stats_seasons = set(stats["season"].dropna().astype(int).unique())
    mvp_in_range  = mvp[mvp["season"].isin(stats_seasons)] if not mvp.empty else pd.DataFrame()

    df = stats.copy()
    df["season"] = df["season"].astype(int)

    # Compute team-level supporting-cast aggregates from the FULL stats data
    # (before any candidate filtering, so teammates are not dropped).
    team_cast = _compute_supporting_cast(df)

    # Merge team win records
    if not teams.empty:
        teams["season"] = teams["season"].astype(int)
        df = df.merge(teams[["season", "team", "wins", "win_pct"]],
                      on=["season", "team"], how="left")

    # Merge MVP labels
    if not mvp_in_range.empty:
        df = df.merge(mvp_in_range[["season", "player", "mvp_winner"]],
                      on=["season", "player"], how="left")
    df["mvp_winner"] = df.get("mvp_winner", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Drop players with essentially no production. Include defensive and kicking
    # stats so that defenders (Alan Page '71, LT '86) and kickers (Moseley '82)
    # — all MVP winners — are NOT filtered out. An MVP winner is always kept
    # regardless, as a safeguard.
    key_stat_cols = [c for c in ["pass_yds", "rush_yds", "rec_yds", "fantasy_pts",
                                  "pass_td", "rush_td", "rec_td",
                                  "def_sacks", "def_int", "def_tds",
                                  "fg_made", "pat_made"] if c in df.columns]
    if key_stat_cols:
        has_production = df[key_stat_cols].fillna(0).abs().sum(axis=1) > 0
        keep = has_production | (df["mvp_winner"] == 1)
        df = df[keep].reset_index(drop=True)

    # Run the shared feature-engineering pipeline
    df = engineer_features(df, team_cast=team_cast, use_weekly=True)

    out_path = PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved features.csv ({len(df)} rows, {len(df.columns)} columns)")
    print(f"  Seasons: {sorted(df['season'].unique())[:3]} ... {sorted(df['season'].unique())[-3:]}")
    print(f"  MVP winners labelled: {df['mvp_winner'].sum()}")
    return df


def engineer_features(df: pd.DataFrame, team_cast: pd.DataFrame = None,
                      use_weekly: bool = True) -> pd.DataFrame:
    """
    Shared feature-engineering pipeline used by BOTH training and prediction,
    so features are computed identically in both paths.

    df         : player-season rows (must have season, team, player, position,
                 and raw box-score stat columns). For prediction, pass the full
                 candidate pool for the season(s) so the relative features
                 (league z-scores, ranks, offense share) are computed correctly.
    team_cast  : precomputed supporting-cast team aggregates. If None, computed
                 from df itself (fine when df already holds the full pool).
    use_weekly : whether to add late-season trend features from the weekly file.
                 Prediction without weekly data should pass False (those features
                 default to 0).
    """
    if team_cast is None:
        team_cast = _compute_supporting_cast(df)

    df = _add_base_features(df)
    df = _flag_games_played(df)
    df = _add_team_success_features(df)
    df = _add_league_comparison_features(df)
    df = _add_position_peer_features(df)
    if use_weekly:
        df = _add_late_season_features(df)
    else:
        df = _ensure_late_season_columns(df)
    df = _add_milestone_features(df)
    df = _add_supporting_cast_features(df, team_cast)
    df = _add_interaction_features(df)
    df = _add_position_prior(df)
    return df


def _ensure_late_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add late-season feature columns as 0 when weekly data isn't available."""
    for col in ["late_fantasy_pg", "early_fantasy_pg", "late_vs_early_fantasy",
                "late_pass_epa_pg", "late_rush_yds_pg", "late_season_trend",
                "strong_finish"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ---------------------------------------------------------------------------
# Base feature engineering
# ---------------------------------------------------------------------------

def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game rates, passer rating, combo stats, position flags."""

    # Total touchdowns
    td_cols = [c for c in ["pass_td", "rush_td", "rec_td", "special_teams_tds"] if c in df.columns]
    df["total_td"] = df[td_cols].fillna(0).sum(axis=1)

    # Yards from scrimmage
    scrimmage_cols = [c for c in ["rush_yds", "rec_yds"] if c in df.columns]
    df["scrimmage_yds"] = df[scrimmage_cols].fillna(0).sum(axis=1)

    # Position-adjusted fantasy score — all TDs worth 6pts regardless of position.
    # Standard fantasy scoring gives QBs only 4pts per passing TD, which artificially
    # inflates QB fantasy totals vs RBs/WRs. This levels the playing field.
    pass_td  = df["pass_td"].fillna(0)  if "pass_td"  in df.columns else 0
    rush_td  = df["rush_td"].fillna(0)  if "rush_td"  in df.columns else 0
    rec_td   = df["rec_td"].fillna(0)   if "rec_td"   in df.columns else 0
    pass_yds = df["pass_yds"].fillna(0) if "pass_yds" in df.columns else 0
    rush_yds = df["rush_yds"].fillna(0) if "rush_yds" in df.columns else 0
    rec_yds  = df["rec_yds"].fillna(0)  if "rec_yds"  in df.columns else 0
    pass_int = df["pass_int"].fillna(0) if "pass_int" in df.columns else 0
    rec      = df["rec"].fillna(0)      if "rec"      in df.columns else 0
    df["fantasy_pts_adj"] = (
        pass_yds * 0.04 +   # 1pt per 25 pass yds
        rush_yds * 0.1  +   # 1pt per 10 rush yds
        rec_yds  * 0.1  +   # 1pt per 10 rec yds
        pass_td  * 6    +   # 6pts all TDs (was 4 for passing)
        rush_td  * 6    +
        rec_td   * 6    +
        rec      * 0.5  +   # half-PPR
        pass_int * -2       # INT penalty
    )

    # Passer rating
    if "pass_rating" not in df.columns and all(
        c in df.columns for c in ["pass_cmp", "pass_att", "pass_yds", "pass_td", "pass_int"]
    ):
        df["pass_rating"] = _calc_passer_rating(df)

    # Completion %
    if "pass_cmp" in df.columns and "pass_att" in df.columns:
        df["pass_cmp_pct"] = (df["pass_cmp"] / df["pass_att"].replace(0, np.nan)).round(4)

    # TD:INT ratio
    if "pass_td" in df.columns and "pass_int" in df.columns:
        df["td_int_ratio"] = df["pass_td"] / df["pass_int"].replace(0, 0.5)

    # Yards per attempt
    if "pass_yds" in df.columns and "pass_att" in df.columns:
        df["pass_ypa"] = (df["pass_yds"] / df["pass_att"].replace(0, np.nan)).round(2)

    # Yards per carry
    if "rush_yds" in df.columns and "rush_att" in df.columns:
        df["rush_ypc"] = (df["rush_yds"] / df["rush_att"].replace(0, np.nan)).round(2)

    # Position flags
    if "position" in df.columns:
        pos = df["position"].str.upper().fillna("")
        df["is_qb"]  = pos.str.contains(r"\bQB\b",  regex=True).astype(int)
        df["is_rb"]  = pos.str.contains(r"\bRB\b|\bHB\b|\bFB\b", regex=True).astype(int)
        df["is_wr"]  = pos.str.contains(r"\bWR\b",  regex=True).astype(int)
        df["is_te"]  = pos.str.contains(r"\bTE\b",  regex=True).astype(int)
        df["is_def"] = pos.str.contains(r"\bDE\b|\bDT\b|\bLB\b|\bCB\b|\bS\b|\bFS\b|\bSS\b", regex=True).astype(int)
        df["is_k"]   = pos.str.contains(r"\bK\b|\bPK\b", regex=True).astype(int)

    # ---- Defensive composite features ----
    # A defensive "impact score" combining the splashy stats voters notice:
    # sacks (1982+), interceptions, defensive TDs. Sacks weighted most heavily
    # since sack leaders are the usual defensive MVP candidates.
    def _dc(name):
        return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)
    df["def_sacks"] = _dc("def_sacks")
    df["def_int"]   = _dc("def_int")
    df["def_tds"]   = _dc("def_tds")
    df["def_impact"] = (
        df["def_sacks"] * 2.0 +   # a sack is the headline defensive play
        df["def_int"]   * 3.0 +   # INTs are rarer and game-changing
        df["def_tds"]   * 6.0     # defensive TD = huge
    )

    # ---- Kicking composite features ----
    df["fg_made"] = _dc("fg_made")
    df["fg_att"]  = _dc("fg_att")
    df["pat_made"] = _dc("pat_made")
    df["fg_pct"] = (df["fg_made"] / df["fg_att"].replace(0, np.nan)).round(4)
    # Kicker scoring points (3 per FG, 1 per PAT) — Moseley led the league in 1982
    df["kick_points"] = df["fg_made"] * 3 + df["pat_made"]

    # Per-game rates
    if "games" in df.columns:
        g = df["games"].replace(0, np.nan)
        for src, dst in [
            ("pass_yds",      "pass_yds_pg"),
            ("pass_td",       "pass_td_pg"),
            ("pass_epa",      "pass_epa_pg"),
            ("rush_yds",      "rush_yds_pg"),
            ("rush_td",       "rush_td_pg"),
            ("rec_yds",       "rec_yds_pg"),
            ("rec_td",        "rec_td_pg"),
            ("total_td",      "total_td_pg"),
            ("scrimmage_yds", "scrimmage_yds_pg"),
            ("fantasy_pts",   "fantasy_pts_pg"),
            ("def_sacks",     "def_sacks_pg"),
            ("def_impact",    "def_impact_pg"),
        ]:
            if src in df.columns:
                df[dst] = (df[src] / g).round(3)

        if "fantasy_pts_adj" in df.columns:
            df["fantasy_pts_adj_pg"] = (df["fantasy_pts_adj"] / g).round(3)

    return df


def _flag_games_played(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag players who played a full season (>= 75% of the season median).
    Part-time and injured players are excluded from peer reference groups
    so they don't drag down the comparison baselines.
    """
    if "games" not in df.columns:
        df["full_season"] = True
        return df
    median_games = df.groupby("season")["games"].transform("median")
    df["full_season"] = df["games"] >= (median_games * MIN_GAMES_FRACTION)
    return df


# ---------------------------------------------------------------------------
# Team success features
# ---------------------------------------------------------------------------

def _add_team_success_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contextualise team performance relative to the rest of the league
    in that season. MVP voters heavily favour players on winning teams.
    """
    if "win_pct" not in df.columns:
        return df

    new_cols = {}

    # Win% percentile vs all teams that season (0 = worst, 1 = best)
    new_cols["team_win_pct_rank"] = df.groupby("season")["win_pct"].rank(pct=True).round(3)

    # Win% z-score vs that season's teams
    season_win_mean = df.groupby("season")["win_pct"].transform("mean")
    season_win_std  = df.groupby("season")["win_pct"].transform("std").replace(0, np.nan)
    new_cols["team_win_zscore"] = ((df["win_pct"] - season_win_mean) / season_win_std).round(3)

    # Playoff-calibre flag: top half of teams by win% that season
    median_win = df.groupby("season")["win_pct"].transform("median")
    new_cols["playoff_team"] = (df["win_pct"] >= median_win).astype(int)

    # Elite team flag: top 25% of teams by win% that season
    p75_win = df.groupby("season")["win_pct"].transform(lambda x: x.quantile(0.75))
    new_cols["elite_team"] = (df["win_pct"] >= p75_win).astype(int)

    # Win% vs average team that season (raw gap, easy to interpret)
    new_cols["win_pct_vs_avg"] = (df["win_pct"] - season_win_mean).round(3)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# League-wide season comparisons
# ---------------------------------------------------------------------------

# Stats ranked across the entire league (all positions, full-season players)
LEAGUE_RANK_STATS = [
    "pass_yds", "pass_td", "pass_epa",
    "pass_yds_pg", "pass_td_pg", "pass_epa_pg",
    "rush_yds", "rush_td", "rush_yds_pg", "rush_td_pg",
    "rec_yds", "rec_td", "rec_yds_pg", "rec_td_pg",
    "total_td", "total_td_pg",
    "scrimmage_yds", "scrimmage_yds_pg",
    "fantasy_pts", "fantasy_pts_pg",
    "fantasy_pts_adj", "fantasy_pts_adj_pg",   # position-equalised scoring
    "pass_rating", "td_int_ratio",
    # Defensive & kicking — so defenders/kickers can be ranked within a season
    "def_sacks", "def_int", "def_impact", "def_impact_pg",
    "fg_made", "fg_pct", "kick_points",
]

# Z-score threshold above which a performance is considered historically dominant.
# At 3.5+ SDs above the mean a player is having a generational season — this
# should signal MVP regardless of position.
DOMINANCE_THRESHOLD = 3.5

# Stats we check for historic dominance (non-QB focused to correct the bias)
DOMINANCE_STATS = [
    "rush_yds", "rush_yds_pg",
    "scrimmage_yds", "scrimmage_yds_pg",
    "total_td",
    "fantasy_pts_adj", "fantasy_pts_adj_pg",
    "pass_yds", "pass_epa",
]


def _add_league_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each key stat, compare every player to the full-season average
    for that season across the whole league.

    Also adds:
      - dominance_score   : count of stats where player is >= DOMINANCE_THRESHOLD SDs above mean
      - historically_dominant : 1 if dominance_score >= 2 (transcendent season signal)
    """
    new_cols = {}
    full_mask = df["full_season"]

    # Reference population for comparisons. Offensive stats are compared only
    # among offensive skill players (QB/RB/WR/TE); defensive & kicking stats
    # among their own kind. This keeps a QB's z-score/rank from being distorted
    # by the hundreds of defenders, linemen and punters on a roster.
    pos_up = df["position"].astype(str).str.upper()
    is_skill = pos_up.str.contains(r"QB|RB|HB|FB|WR|TE", regex=True, na=False)
    is_defkick = pos_up.str.contains(r"\bK\b|PK|DE|DT|LB|CB|\bS\b|FS|SS|NT|DB|DL",
                                     regex=True, na=False)

    DEFKICK_STATS = {"def_sacks", "def_int", "def_impact", "def_impact_pg",
                     "fg_made", "fg_pct", "kick_points"}

    for stat in LEAGUE_RANK_STATS:
        if stat not in df.columns:
            continue

        # Pick the reference population for this stat
        if stat in DEFKICK_STATS:
            ref_mask = full_mask & is_defkick
        else:
            ref_mask = full_mask & is_skill

        mean_map = df[ref_mask].groupby("season")[stat].mean()
        std_map  = df[ref_mask].groupby("season")[stat].std()

        s_mean = df["season"].map(mean_map)
        s_std  = df["season"].map(std_map).replace(0, np.nan)

        new_cols[f"{stat}_league_zscore"] = ((df[stat] - s_mean) / s_std).round(3)
        new_cols[f"{stat}_vs_league_avg"] = (df[stat] - s_mean).round(2)

        # Rank only within the relevant population (others get NaN -> bottom)
        rank_pop = df[ref_mask]
        rank_in_pop = (
            rank_pop.groupby("season")[stat]
            .rank(ascending=False, method="min", na_option="bottom")
        )
        new_cols[f"{stat}_league_rank"] = rank_in_pop.reindex(df.index).fillna(999).astype(int)
        pct_in_pop = rank_pop.groupby("season")[stat].rank(pct=True)
        new_cols[f"{stat}_league_pct"] = pct_in_pop.reindex(df.index).round(3)

    # Build a temporary df with z-scores to compute dominance
    zscore_cols = [f"{s}_league_zscore" for s in DOMINANCE_STATS
                   if f"{s}_league_zscore" in new_cols]
    if zscore_cols:
        zscore_df = pd.DataFrame({c: new_cols[c] for c in zscore_cols}, index=df.index)
        # Count how many dominance stats are >= threshold
        new_cols["dominance_score"] = (zscore_df >= DOMINANCE_THRESHOLD).sum(axis=1)
        # Binary flag: at least 2 stats are historically dominant
        new_cols["historically_dominant"] = (new_cols["dominance_score"] >= 2).astype(int)
        # Continuous version: max z-score across dominance stats (captures peak performance)
        new_cols["peak_dominance_zscore"] = zscore_df.max(axis=1).round(3)

        # Yardage-specific dominance: catches Peterson/Faulk/LT type seasons where
        # rushing/scrimmage yardage is historically freakish even if TDs are modest.
        # Triggered when rush OR scrimmage yards z-score is >= 4.5
        yard_cols = [c for c in ["rush_yds_league_zscore", "scrimmage_yds_league_zscore",
                                  "rush_yds_pg_league_zscore", "scrimmage_yds_pg_league_zscore"]
                     if c in new_cols]
        if yard_cols:
            yard_df = pd.DataFrame({c: new_cols[c] for c in yard_cols}, index=df.index)
            new_cols["yardage_dominant"] = (yard_df >= 4.5).any(axis=1).astype(int)
            new_cols["max_yardage_zscore"] = yard_df.max(axis=1).round(3)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Position peer comparisons
# ---------------------------------------------------------------------------

# Stats compared within position peer groups
POSITION_PEER_STATS = {
    "QB": ["pass_yds", "pass_td", "pass_epa", "pass_rating",
           "td_int_ratio", "pass_yds_pg", "pass_td_pg", "pass_epa_pg",
           "fantasy_pts_adj", "fantasy_pts_adj_pg",
           # QB rushing compared to OTHER QBs — a QB with 800 rush yards is
           # exceptional (Lamar), whereas that's ordinary for an RB. Comparing
           # within the QB pool captures that dual-threat value correctly.
           "rush_yds", "rush_td", "rush_yds_pg"],
    "RB": ["rush_yds", "rush_td", "rush_yds_pg", "rush_ypc",
           "scrimmage_yds", "scrimmage_yds_pg",
           "fantasy_pts_adj", "fantasy_pts_adj_pg"],
    "WR": ["rec_yds", "rec_td", "rec_yds_pg", "targets", "target_share",
           "wopr", "fantasy_pts_adj", "fantasy_pts_adj_pg"],
    "TE": ["rec_yds", "rec_td", "targets", "target_share",
           "fantasy_pts_adj", "fantasy_pts_adj_pg"],
}


def _add_position_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each position group, compare players only against other starters
    at the same position with comparable volume (e.g. QBs with 400+ attempts).

    Produces per stat per position:
      - <stat>_pos_zscore   : z-score vs positional peers (full-volume starters)
      - <stat>_pos_pct      : percentile rank among positional peers
      - <stat>_vs_pos_avg   : raw gap above the positional peer average
      - <stat>_pos_rank     : integer rank among positional peers (1 = best)
    """
    new_cols = {}

    for pos_key, stats in POSITION_PEER_STATS.items():
        # Build position mask (handle positional group variations)
        if pos_key == "QB":
            pos_mask = df["position"].str.upper().str.contains(r"\bQB\b", regex=True, na=False)
        elif pos_key == "RB":
            pos_mask = df["position"].str.upper().str.contains(r"\bRB\b|\bHB\b|\bFB\b", regex=True, na=False)
        elif pos_key == "WR":
            pos_mask = df["position"].str.upper().str.contains(r"\bWR\b", regex=True, na=False)
        elif pos_key == "TE":
            pos_mask = df["position"].str.upper().str.contains(r"\bTE\b", regex=True, na=False)
        else:
            continue

        # Volume filter: only include players who meet the starter threshold
        vol_filter = pd.Series(True, index=df.index)
        if pos_key in VOLUME_THRESHOLDS:
            for vol_col, threshold in VOLUME_THRESHOLDS[pos_key].items():
                if vol_col in df.columns:
                    vol_filter = vol_filter & (df[vol_col].fillna(0) >= threshold)

        # Full-season AND positional starter
        peer_mask = pos_mask & df["full_season"] & vol_filter

        for stat in stats:
            if stat not in df.columns:
                continue

            # Compute peer group mean/std per season
            peer_mean_map = df[peer_mask].groupby("season")[stat].mean()
            peer_std_map  = df[peer_mask].groupby("season")[stat].std()

            s_mean = df["season"].map(peer_mean_map)
            s_std  = df["season"].map(peer_std_map).replace(0, np.nan)

            col_prefix = f"{stat}_{pos_key.lower()}"

            # Z-score vs positional peers
            new_cols[f"{col_prefix}_pos_zscore"] = (
                ((df[stat] - s_mean) / s_std)
                .where(pos_mask, np.nan)   # only meaningful for that position
                .round(3)
            )

            # Raw gap above positional peer average
            new_cols[f"{col_prefix}_vs_pos_avg"] = (
                (df[stat] - s_mean)
                .where(pos_mask, np.nan)
                .round(2)
            )

            # Percentile within position group
            def _pos_pct_rank(grp):
                return grp.rank(pct=True)

            pos_pct = (
                df[pos_mask]
                .groupby("season")[stat]
                .rank(pct=True)
                .round(3)
            )
            new_cols[f"{col_prefix}_pos_pct"] = pos_pct.reindex(df.index)

            # Integer rank within position group
            pos_rank = (
                df[pos_mask]
                .groupby("season")[stat]
                .rank(ascending=False, method="min", na_option="bottom")
                .astype(float)
            )
            new_cols[f"{col_prefix}_pos_rank"] = pos_rank.reindex(df.index)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Late-season performance trends
# ---------------------------------------------------------------------------

# How many weeks count as "late season"
LATE_WEEKS = 6   # final 6 weeks of regular season


def _add_late_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the weekly stats file and compute late-season performance features:

      - late_fantasy_pg         : avg fantasy pts per game in final LATE_WEEKS weeks
      - early_fantasy_pg        : avg fantasy pts per game in first (season - LATE_WEEKS) weeks
      - late_vs_early_fantasy   : late minus early (positive = finishing strong)
      - late_pass_epa_pg        : avg passing EPA per game late season
      - late_rush_yds_pg        : avg rushing yards per game late season
      - late_season_trend       : z-score of late_vs_early within that season
                                  (how much stronger/weaker vs other players' trends)
      - strong_finish           : 1 if late_vs_early_fantasy > 0 AND late_fantasy_pg
                                  is above the season median late fantasy
    """
    weekly_path = RAW_DIR / "player_stats_weekly.csv"
    if not weekly_path.exists():
        print("  WARN: player_stats_weekly.csv not found — skipping late-season features.")
        return df

    weekly = pd.read_csv(weekly_path, low_memory=False)

    # Normalise column names
    rename = {
        "player_display_name": "player",
        "recent_team": "team",
        "passing_yards": "pass_yds", "passing_tds": "pass_td",
        "passing_epa": "pass_epa",
        "rushing_yards": "rush_yds", "rushing_tds": "rush_td",
        "receiving_yards": "rec_yds", "receiving_tds": "rec_td",
        "fantasy_points": "fantasy_pts",
    }
    weekly = weekly.rename(columns={k: v for k, v in rename.items() if k in weekly.columns})

    if "player" not in weekly.columns:
        print("  WARN: weekly data missing player column — skipping late-season features.")
        return df

    # Drop duplicate player-game rows (legacy source can carry dupes) so the
    # late-season averages aren't distorted. Key on player_id when available.
    dedup_key = "player_id" if "player_id" in weekly.columns else "player"
    weekly = weekly.drop_duplicates(subset=[dedup_key, "season", "week"]).copy()

    # Position-adjusted fantasy for weekly rows too.
    # Build each component safely in case a column is missing.
    def _col(name):
        if name in weekly.columns:
            return pd.to_numeric(weekly[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=weekly.index)

    weekly["fantasy_pts_adj"] = (
        _col("pass_yds") * 0.04 +
        _col("rush_yds") * 0.1  +
        _col("rec_yds")  * 0.1  +
        _col("pass_td")  * 6    +
        _col("rush_td")  * 6    +
        _col("rec_td")   * 6    +
        _col("pass_int") * -2
    )

    new_cols = {}

    for (season, player), grp in weekly.groupby(["season", "player"]):
        grp = grp.sort_values("week")
        max_week = grp["week"].max()
        cutoff   = max_week - LATE_WEEKS + 1

        late  = grp[grp["week"] >= cutoff]
        early = grp[grp["week"] <  cutoff]

        late_fp  = late["fantasy_pts_adj"].mean()  if len(late)  > 0 else np.nan
        early_fp = early["fantasy_pts_adj"].mean() if len(early) > 0 else np.nan
        late_pass_epa = late["pass_epa"].mean()  if "pass_epa"  in late.columns and len(late) > 0 else np.nan
        late_rush_yds = late["rush_yds"].mean()  if "rush_yds"  in late.columns and len(late) > 0 else np.nan

        key = (season, player)
        new_cols[key] = {
            "late_fantasy_pg":       round(late_fp,  2) if not np.isnan(late_fp)  else np.nan,
            "early_fantasy_pg":      round(early_fp, 2) if not np.isnan(early_fp) else np.nan,
            "late_vs_early_fantasy": round(late_fp - early_fp, 2)
                                     if not (np.isnan(late_fp) or np.isnan(early_fp)) else np.nan,
            "late_pass_epa_pg":      round(late_pass_epa, 3) if not np.isnan(late_pass_epa) else np.nan,
            "late_rush_yds_pg":      round(late_rush_yds, 1) if not np.isnan(late_rush_yds) else np.nan,
        }

    trend_df = pd.DataFrame(new_cols).T
    trend_df.index = pd.MultiIndex.from_tuples(trend_df.index, names=["season", "player"])
    trend_df = trend_df.reset_index()
    trend_df["season"] = trend_df["season"].astype(int)

    # Merge onto main df
    df = df.merge(trend_df, on=["season", "player"], how="left")

    # Late-season trend z-score within season — how strong a finish relative to peers
    if "late_vs_early_fantasy" in df.columns:
        df["late_season_trend"] = df.groupby("season")["late_vs_early_fantasy"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        ).round(3)

    # Strong finish flag: improving AND above median late-season pace
    if "late_vs_early_fantasy" in df.columns and "late_fantasy_pg" in df.columns:
        median_late = df.groupby("season")["late_fantasy_pg"].transform("median")
        df["strong_finish"] = (
            (df["late_vs_early_fantasy"] > 0) &
            (df["late_fantasy_pg"] >= median_late)
        ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Milestone features
# ---------------------------------------------------------------------------

def _add_milestone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags for hitting the round-number statistical milestones that
    historically correlate heavily with MVP voting. Voters fixate on these
    headline marks (e.g. 2,000 rushing yards, 5,000 passing yards, 50 TDs).

    Also adds a 'milestone_count' tallying how many a player hit, and a
    graded 'milestone_score' weighting the rarer ones more heavily.
    """
    new_cols = {}

    def col(name):
        return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)

    pass_yds = col("pass_yds")
    pass_td  = col("pass_td")
    rush_yds = col("rush_yds")
    rush_td  = col("rush_td")
    rec_yds  = col("rec_yds")
    rec_td   = col("rec_td")
    total_td = col("total_td")
    scrimmage = col("scrimmage_yds")

    # Passing milestones (weight = rarity/prestige)
    new_cols["ms_pass_4000"] = (pass_yds >= 4000).astype(int)
    new_cols["ms_pass_4500"] = (pass_yds >= 4500).astype(int)
    new_cols["ms_pass_5000"] = (pass_yds >= 5000).astype(int)   # very rare, huge signal
    new_cols["ms_pass_td_30"] = (pass_td >= 30).astype(int)
    new_cols["ms_pass_td_40"] = (pass_td >= 40).astype(int)
    new_cols["ms_pass_td_50"] = (pass_td >= 50).astype(int)     # historic (Manning '13, Mahomes '18)

    # Rushing milestones
    new_cols["ms_rush_1500"] = (rush_yds >= 1500).astype(int)
    new_cols["ms_rush_1800"] = (rush_yds >= 1800).astype(int)
    new_cols["ms_rush_2000"] = (rush_yds >= 2000).astype(int)   # the magic number
    new_cols["ms_rush_td_15"] = (rush_td >= 15).astype(int)
    new_cols["ms_rush_td_20"] = (rush_td >= 20).astype(int)

    # Receiving milestones
    new_cols["ms_rec_1500"] = (rec_yds >= 1500).astype(int)
    new_cols["ms_rec_1800"] = (rec_yds >= 1800).astype(int)
    new_cols["ms_rec_td_15"] = (rec_td >= 15).astype(int)

    # Total TD milestones
    new_cols["ms_total_td_20"] = (total_td >= 20).astype(int)
    new_cols["ms_total_td_25"] = (total_td >= 25).astype(int)
    new_cols["ms_total_td_30"] = (total_td >= 30).astype(int)

    # Scrimmage yards milestone (all-purpose backs)
    new_cols["ms_scrimmage_2000"] = (scrimmage >= 2000).astype(int)
    new_cols["ms_scrimmage_2500"] = (scrimmage >= 2500).astype(int)

    # ---- League-leader flags ----
    # Leading the league in a headline stat is a distinct MVP signal that a
    # smooth percentile rank understates. Voters fixate on "led the NFL in
    # passing TDs / yards / rushing yards". These require the *_league_rank
    # columns, which are computed earlier in the pipeline.
    def led(rank_col, within=1):
        if rank_col in df.columns:
            return (df[rank_col] <= within).astype(int)
        return pd.Series(0, index=df.index)

    new_cols["led_pass_td"]   = led("pass_td_league_rank")
    new_cols["led_pass_yds"]  = led("pass_yds_league_rank")
    new_cols["led_rush_yds"]  = led("rush_yds_league_rank")
    new_cols["led_rush_td"]   = led("rush_td_league_rank")
    new_cols["led_total_td"]  = led("total_td_league_rank")
    new_cols["led_scrimmage"] = led("scrimmage_yds_league_rank")
    new_cols["top3_pass_td"]  = led("pass_td_league_rank", 3)
    new_cols["top3_pass_yds"] = led("pass_yds_league_rank", 3)
    new_cols["top3_rush_yds"] = led("rush_yds_league_rank", 3)

    # "Passing double/triple" — leading in multiple passing categories at once
    # (a very strong QB-MVP signal, e.g. Stafford 2025 led yards AND TDs)
    new_cols["pass_cats_led"] = new_cols["led_pass_td"] + new_cols["led_pass_yds"]
    new_cols["rush_cats_led"] = new_cols["led_rush_yds"] + new_cols["led_rush_td"]

    # Count of milestones hit
    ms_df = pd.DataFrame(new_cols, index=df.index)
    new_cols["milestone_count"] = ms_df.sum(axis=1)

    # Weighted score — rarer milestones worth more
    weights = {
        "ms_pass_5000": 4, "ms_pass_td_50": 4, "ms_rush_2000": 5, "ms_scrimmage_2500": 4,
        "ms_pass_4500": 2, "ms_pass_td_40": 2, "ms_rush_1800": 3, "ms_rush_td_20": 3,
        "ms_rec_1800": 3, "ms_total_td_30": 3, "ms_scrimmage_2000": 2,
        "ms_pass_4000": 1, "ms_pass_td_30": 1, "ms_rush_1500": 1, "ms_rush_td_15": 1,
        "ms_rec_1500": 1, "ms_rec_td_15": 1, "ms_total_td_20": 1, "ms_total_td_25": 2,
    }
    score = pd.Series(0.0, index=df.index)
    for k, w in weights.items():
        if k in new_cols:
            score = score + new_cols[k] * w
    new_cols["milestone_score"] = score

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Supporting cast — carrying vs being carried
# ---------------------------------------------------------------------------

def _compute_supporting_cast(stats: pd.DataFrame) -> pd.DataFrame:
    """
    From the full (unfiltered) stats, compute team-season aggregates used to
    measure how much help each player had. Returns a per-(season, team) frame.
    """
    def col(name):
        return stats[name].fillna(0) if name in stats.columns else pd.Series(0.0, index=stats.index)

    s = pd.DataFrame({
        "season": stats["season"].astype(int),
        "team": stats["team"],
        "player": stats["player"],
        "pass_yds": col("pass_yds"), "pass_td": col("pass_td"),
        "rush_yds": col("rush_yds"), "rush_td": col("rush_td"),
        "rec_yds":  col("rec_yds"),  "rec_td":  col("rec_td"),
    })
    # A team's total offensive output. Passing yards and rushing yards together
    # roughly equal total team yards (receiving yards ≈ passing yards, so we don't
    # double count — team yards = pass_yds + rush_yds summed over the roster's
    # passers and rushers).
    team = s.groupby(["season", "team"]).agg(
        team_pass_yds=("pass_yds", "sum"),
        team_pass_td=("pass_td", "sum"),
        team_rush_yds=("rush_yds", "sum"),
        team_rush_td=("rush_td", "sum"),
        team_rec_yds=("rec_yds", "sum"),
        team_rec_td=("rec_td", "sum"),
    ).reset_index()
    team["team_total_yds"] = team["team_pass_yds"] + team["team_rush_yds"]
    team["team_total_td"] = (team["team_pass_td"] + team["team_rush_td"])

    # Top non-QB receiver yards and top RB rush yards per team (best weapon)
    recs = s.groupby(["season", "team"]).apply(
        lambda g: pd.Series({
            "best_receiver_yds": g["rec_yds"].max(),
            "best_rusher_yds":   g["rush_yds"].max(),
            "second_receiver_yds": g["rec_yds"].nlargest(2).iloc[-1] if (g["rec_yds"] > 0).sum() >= 2 else 0.0,
        })
    ).reset_index()

    return team.merge(recs, on=["season", "team"], how="left")


def _add_supporting_cast_features(df: pd.DataFrame, team_cast: pd.DataFrame) -> pd.DataFrame:
    """
    Add features describing a player's supporting cast and, crucially, the
    SHARE of their team's offense they personally account for. A high share
    means the player is "carrying" the team; a low share means they are being
    helped by strong teammates.
    """
    df = df.merge(team_cast, on=["season", "team"], how="left")

    def c(name):
        return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)

    team_yds = df["team_total_yds"].replace(0, np.nan)
    team_td  = df["team_total_td"].replace(0, np.nan)

    # ---- Share of team offense (the core carrying metric) ----
    # A player's own offensive yards = their passing OR their scrimmage yards.
    # For QBs this is passing yards; for skill players it's rushing+receiving.
    own_pass_yds = c("pass_yds")
    own_scrim_yds = c("rush_yds") + c("rec_yds")

    # QB share = passing yards / team total yards
    df["qb_offense_share"] = (own_pass_yds / team_yds).round(3)
    # Skill player share = their scrimmage yards / team total yards
    df["skill_offense_share"] = (own_scrim_yds / team_yds).round(3)

    # Unified: use the player's dominant contribution type
    is_qb = df.get("is_qb", pd.Series(0, index=df.index)) == 1
    df["offense_share"] = np.where(is_qb, df["qb_offense_share"], df["skill_offense_share"]).round(3)

    # Share of team touchdowns the player accounted for
    own_td = np.where(is_qb, c("pass_td"), c("rush_td") + c("rec_td"))
    df["td_share"] = (own_td / team_td).round(3)

    # ---- Supporting cast quality ----
    # For a QB: how good were their best weapons? Strong weapons = more help.
    df["best_weapon_yds"] = df["best_receiver_yds"].fillna(0)
    df["weapon_support"] = (df["best_receiver_yds"].fillna(0) +
                            df["second_receiver_yds"].fillna(0)).round(0)

    # For a skill player: how good was the rest of the offense around them?
    # Team yards excluding the player's own contribution.
    df["supporting_yds"] = (df["team_total_yds"].fillna(0) - own_scrim_yds).round(0)

    # ---- "Carry index": offense share relative to peers at the same position ----
    # Standardise offense_share within (season, position-group) so we know who
    # was carrying the heaviest load for their role.
    if "position" in df.columns:
        grp_keys = ["season"]
        share_mean = df.groupby(grp_keys)["offense_share"].transform("mean")
        share_std  = df.groupby(grp_keys)["offense_share"].transform("std").replace(0, np.nan)
        df["carry_index"] = ((df["offense_share"] - share_mean) / share_std).round(3)

    # Clean up intermediate team columns we don't need as features
    drop = ["team_pass_yds", "team_pass_td", "team_rush_yds", "team_rush_td",
            "team_rec_yds", "team_rec_td", "best_receiver_yds", "best_rusher_yds",
            "second_receiver_yds"]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit interaction features targeting the remaining model blind spots:

    1. yardage_dominant × team_win_zscore
       A historically dominant rusher on a winning team should rank very high
       regardless of their EPA (which is NaN for RBs). Fixes Faulk 2000,
       Peterson 2012, LT type seasons.

    2. late_season_trend × team_win_zscore
       A player finishing strong AND on a good team — the exact Lamar 2023
       / Allen 2024 signal. Also helps Rodgers 2020/2021.

    3. fading_top_epa penalty
       A QB who leads in EPA but whose late_season_trend is strongly negative
       is likely to lose votes in a close race. Captures Brady 2016 fading
       hard while Ryan was more consistent.

    4. fantasy_adj_rank_1 flag
       Being ranked #1 in position-adjusted fantasy in the whole league is
       historically a very strong MVP signal regardless of position.
       Helps Alexander 2005, Faulk 2000.

    5. strong_team_non_qb
       A non-QB on an elite team with dominant yardage — the specific
       combination that defines the handful of non-QB MVP winners.
    """
    new_cols = {}

    # 1. Yardage dominance × team quality
    if "yardage_dominant" in df.columns and "team_win_zscore" in df.columns:
        new_cols["yardage_dom_x_team"] = (
            df["yardage_dominant"] * df["team_win_zscore"].fillna(0)
        ).round(3)

    # Also: raw yardage z-score × win% (continuous version, not just binary flag)
    if "max_yardage_zscore" in df.columns and "team_win_zscore" in df.columns:
        new_cols["yardage_zscore_x_team"] = (
            df["max_yardage_zscore"].fillna(0) * df["team_win_zscore"].fillna(0)
        ).round(3)

    # 2. Late-season trend × team quality
    if "late_season_trend" in df.columns and "team_win_zscore" in df.columns:
        new_cols["late_trend_x_team"] = (
            df["late_season_trend"].fillna(0) * df["team_win_zscore"].fillna(0)
        ).round(3)

    # Also: strong finish AND good team (binary)
    if "strong_finish" in df.columns and "elite_team" in df.columns:
        new_cols["strong_finish_elite_team"] = (
            df["strong_finish"] * df["elite_team"]
        ).astype(int)

    # 3. Fading top-EPA penalty — high EPA but strongly negative late trend
    #    A large negative value here means the player faded badly despite good EPA
    if "pass_epa_league_zscore" in df.columns and "late_season_trend" in df.columns:
        new_cols["epa_x_late_trend"] = (
            df["pass_epa_league_zscore"].fillna(0) * df["late_season_trend"].fillna(0)
        ).round(3)

    # 4. Fantasy adj rank #1 flag
    if "fantasy_pts_adj_league_rank" in df.columns:
        new_cols["fantasy_adj_rank1"] = (
            df["fantasy_pts_adj_league_rank"] == 1
        ).astype(int)

    # 5. Strong-team non-QB with dominant yardage
    if all(c in df.columns for c in ["is_qb", "elite_team", "yardage_dominant"]):
        new_cols["nonqb_elite_yardage"] = (
            (1 - df["is_qb"]) * df["elite_team"] * df["yardage_dominant"]
        ).astype(int)

    # 6. Late fantasy pace × adjusted fantasy z-score
    #    Players who are both statistically dominant AND finishing strong
    if "late_fantasy_pg" in df.columns and "fantasy_pts_adj_league_zscore" in df.columns:
        new_cols["late_pace_x_adj_zscore"] = (
            df["late_fantasy_pg"].fillna(0) *
            df["fantasy_pts_adj_league_zscore"].fillna(0)
        ).round(3)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Position prior
# ---------------------------------------------------------------------------

def _add_position_prior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small historical prior for how often each position wins MVP.
    The model weights this itself — kept low enough that a dominant
    non-QB season (LT 2006, AP 2012) can still come out on top.
    """
    if "position" not in df.columns:
        df["pos_prior"] = DEFAULT_PRIOR
        return df

    def _prior(pos):
        if pd.isna(pos):
            return DEFAULT_PRIOR
        p = str(pos).upper().strip()
        for key, val in POSITION_PRIOR.items():
            if key in p:
                return val
        return DEFAULT_PRIOR

    df["pos_prior"] = df["position"].apply(_prior)
    return df


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _calc_passer_rating(df: pd.DataFrame) -> pd.Series:
    """Official NFL passer rating formula."""
    att = df["pass_att"].replace(0, np.nan)
    a = ((df["pass_cmp"] / att) - 0.3) * 5
    b = ((df["pass_yds"] / att) - 3) * 0.25
    c = (df["pass_td"] / att) * 20
    d = 2.375 - ((df["pass_int"] / att) * 25)
    return ((a.clip(0, 2.375) + b.clip(0, 2.375) +
             c.clip(0, 2.375) + d.clip(0, 2.375)) / 6 * 100).round(1)


if __name__ == "__main__":
    df = build_features()
    print("\nSample MVP seasons:")
    show_cols = [c for c in [
        "season", "player", "position", "team",
        "pass_yds", "rush_yds", "total_td", "win_pct",
        "team_win_pct_rank", "elite_team",
        "fantasy_pts_league_rank", "pass_epa_qb_pos_zscore", "rush_yds_rb_pos_zscore",
    ] if c in df.columns]
    mvp_rows = df[df["mvp_winner"] == 1][show_cols]
    print(mvp_rows.sort_values("season").to_string(index=False))
