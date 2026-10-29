"""
features_spec.py
----------------
Defines the model's feature sets and position-track assignment. This is the
single source of truth for WHICH engineered columns the model uses and HOW
players are grouped by position. The ranker (model_ranker.py) imports from here.

Tracks:
  qb      : quarterbacks
  skill   : RB / WR / TE
  (offense = qb + skill, used when ranking all offensive candidates together)
  defkick : defensive players + kickers — out of scope for the modern model
  none    : offensive line, punters, snappers — never MVP candidates
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Base feature groups
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

SHARED_CONTEXT = [
    "games", "full_season",
    "wins", "win_pct", "team_win_zscore", "team_win_pct_rank",
    "elite_team", "playoff_team", "win_pct_vs_avg",
    "pos_prior",
]

# Shared offensive production + comparison features (used by both qb and skill)
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
# is a strong, non-redundant MVP signal (+8pts top-1); the other leader flags
# were noise or collinear with existing rank features, so only these are kept.
QB_LEADER_FLAGS = ["led_pass_yds"]
SKILL_LEADER_FLAGS = ["led_rush_yds", "led_scrimmage"]


# ---------------------------------------------------------------------------
# Track feature sets
# ---------------------------------------------------------------------------

# QB track: passing-led, heavy team-success emphasis, plus QB rushing
QB_TRACK_FEATURES = (PASSING_FEATURES + [
    "rush_yds", "rush_td", "rush_yds_pg",
    "rush_yds_qb_pos_zscore", "rush_td_qb_pos_zscore", "rush_yds_qb_pos_pct",
    "qb_offense_share", "weapon_support", "best_weapon_yds", "supporting_yds",
    "late_pass_epa_pg",
] + QB_LEADER_FLAGS + _SHARED_OFFENSE + SHARED_CONTEXT)

# Skill track (RB/WR/TE): rushing/receiving-led, carry emphasis
SKILL_TRACK_FEATURES = (RUSHING_FEATURES + RECEIVING_FEATURES + [
    "is_rb", "is_wr", "is_te",
    "skill_offense_share", "supporting_yds",
    "late_rush_yds_pg",
] + SKILL_LEADER_FLAGS + _SHARED_OFFENSE + SHARED_CONTEXT)

TRACK_FEATURES = {
    "qb": QB_TRACK_FEATURES,
    "skill": SKILL_TRACK_FEATURES,
}


# ---------------------------------------------------------------------------
# Position → track
# ---------------------------------------------------------------------------

def assign_track(position) -> str:
    """
    Map a position to its MVP contest track. QBs and skill players are split
    because the MVP formula differs by position: QBs are gated hard by team
    success (every modern QB MVP was on an elite team), while RBs/receivers
    can win by carrying a heavier individual load on a weaker team.
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


def get_feature_cols(df: pd.DataFrame, track: str) -> list:
    """Return the track's feature columns that exist in df (de-duplicated)."""
    base = TRACK_FEATURES[track]
    seen, cols = set(), []
    for c in base:
        if c in df.columns and c not in seen:
            seen.add(c)
            cols.append(c)
    return cols
