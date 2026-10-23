"""
ingest_local.py
---------------
Processes the large local game-by-game parquet file
(NFL_All_Player_Game_Data_1950_2026.parquet) into the same raw CSV format
the rest of the pipeline expects:

  - player_stats.csv         : season totals per player
  - player_stats_weekly.csv  : week-by-week rows (late-season features)
  - team_records.csv         : team win/loss records per season

This replaces the nflverse download for the expanded 1957–2024 dataset.
Pre-1999 seasons use traditional stats (no EPA, which didn't exist yet).
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_PARQUET = PROJECT_ROOT / "NFL_All_Player_Game_Data_1950_2026.parquet"

# AP MVP era: 1957 onward. Include through 2026 (current season, partial).
INGEST_START = 1957
INGEST_END   = 2026


def _regular_season_games(season: int) -> int:
    """
    Number of regular-season games by era. Weeks beyond this in the legacy
    (UNKNOWN season_type) data are playoff games and must be dropped.
      1950–1960: 12 games
      1961–1977: 14 games
      1978–1989: 16 games (strike years 1982/1987 had fewer, handled by cap)
      1990–2020: 16 games
      2021+:     17 games
    """
    if season <= 1960:
        return 12
    if season <= 1977:
        return 14
    if season <= 2020:
        return 16
    return 17


def _strip_legacy_playoffs(legacy: pd.DataFrame) -> pd.DataFrame:
    """
    Remove playoff rows from the legacy data. Since season_type is UNKNOWN,
    we identify playoffs by week number exceeding the era's regular-season
    length. A small buffer (+1) tolerates bye-week numbering quirks.
    """
    if legacy.empty:
        return legacy
    caps = legacy["season"].map(_regular_season_games)
    wk = pd.to_numeric(legacy["week"], errors="coerce")
    # keep weeks within the regular-season cap (NaN weeks kept — rare)
    keep = wk.isna() | (wk <= caps)
    return legacy[keep].copy()


def _coalesce(df: pd.DataFrame, modern: str, legacy: str) -> pd.Series:
    """
    Combine a modern column with its legacy counterpart.
    Uses the modern value where present, falls back to legacy.
    Pre-1999 rows generally only have legacy populated.
    """
    m = df[modern] if modern in df.columns else pd.Series(np.nan, index=df.index)
    l = df[legacy] if legacy in df.columns else pd.Series(np.nan, index=df.index)
    m = pd.to_numeric(m, errors="coerce")
    l = pd.to_numeric(l, errors="coerce")
    return m.fillna(l)


def load_raw() -> pd.DataFrame:
    if not LOCAL_PARQUET.exists():
        raise FileNotFoundError(
            f"Local data file not found: {LOCAL_PARQUET.name}. "
            "Place it in the project root."
        )
    print(f"Loading {LOCAL_PARQUET.name} ...")
    df = pd.read_parquet(LOCAL_PARQUET)
    print(f"  {len(df):,} rows loaded")
    return df


def build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean weekly table with coalesced modern+legacy stats.
    One row per player per game.
    """
    # Regular season (older rows are tagged UNKNOWN but are regular-season games)
    df = df[(df["season"] >= INGEST_START) & (df["season"] <= INGEST_END)].copy()

    # ------------------------------------------------------------------
    # De-duplicate the two data sources.
    #   - nflverse_weekly_player_stats: 1999+, cleanly tags REG vs POST, has EPA
    #   - legacy_raw_player_game:       1950-2017, everything tagged UNKNOWN
    #                                   (regular + playoff games mixed together)
    # For 1999+ we trust ONLY nflverse (filtered to REG) to avoid double-counting
    # and to drop playoff games. For pre-1999 we use the legacy source and strip
    # playoff weeks by capping at the era's regular-season game count.
    # ------------------------------------------------------------------
    NFLVERSE_FIRST = 1999

    nfl = df[(df["data_source"] == "nflverse_weekly_player_stats") &
             (df["season"] >= NFLVERSE_FIRST) &
             (df["season_type"] == "REG")].copy()

    legacy = df[(df["data_source"] == "legacy_raw_player_game") &
                (df["season"] < NFLVERSE_FIRST)].copy()

    # Strip playoff weeks from the legacy data using era-appropriate caps.
    legacy = _strip_legacy_playoffs(legacy)

    df = pd.concat([nfl, legacy], ignore_index=True)

    out = pd.DataFrame()
    out["season"] = df["season"].astype(int)
    out["week"]   = pd.to_numeric(df["week"], errors="coerce")
    out["player"] = df["player_display_name"].fillna(df["player_name"])
    out["player_id"] = df["player_id"]
    out["team"]     = df["team"]
    out["position"] = df["position"].fillna(df.get("player_meta_position")).apply(normalise_position)
    out["game_won"] = df["game_won"]

    # Passing
    out["pass_cmp"]  = _coalesce(df, "completions", "legacy_passing_completions")
    out["pass_att"]  = _coalesce(df, "attempts", "legacy_passing_attempts")
    out["pass_yds"]  = _coalesce(df, "passing_yards", "legacy_passing_yards")
    out["pass_td"]   = _coalesce(df, "passing_tds", "legacy_passing_touchdowns")
    out["pass_int"]  = _coalesce(df, "passing_interceptions", "legacy_passing_interceptions")
    out["pass_epa"]  = pd.to_numeric(df.get("passing_epa"), errors="coerce")

    # Rushing
    out["rush_att"]  = _coalesce(df, "carries", "legacy_rushing_attempts")
    out["rush_yds"]  = _coalesce(df, "rushing_yards", "legacy_rushing_yards")
    out["rush_td"]   = _coalesce(df, "rushing_tds", "legacy_rushing_touchdowns")
    out["rush_epa"]  = pd.to_numeric(df.get("rushing_epa"), errors="coerce")

    # Receiving
    out["rec"]       = _coalesce(df, "receptions", "legacy_receiving_receptions")
    out["targets"]   = _coalesce(df, "targets", "legacy_receiving_targets")
    out["rec_yds"]   = _coalesce(df, "receiving_yards", "legacy_receiving_yards")
    out["rec_td"]    = _coalesce(df, "receiving_tds", "legacy_receiving_touchdowns")
    out["rec_epa"]   = pd.to_numeric(df.get("receiving_epa"), errors="coerce")

    # Advanced receiving (modern only)
    out["target_share"] = pd.to_numeric(df.get("target_share"), errors="coerce")
    out["wopr"]         = pd.to_numeric(df.get("wopr"), errors="coerce")
    out["racr"]         = pd.to_numeric(df.get("racr"), errors="coerce")
    out["pacr"]         = pd.to_numeric(df.get("pacr"), errors="coerce")

    # Defense
    out["def_sacks"]  = _coalesce(df, "def_sacks", "legacy_defense_sacks")
    out["def_int"]    = _coalesce(df, "def_interceptions", "legacy_defense_interceptions")
    out["def_tds"]    = pd.to_numeric(df.get("def_tds"), errors="coerce")

    # Kicking
    out["fg_made"]   = _coalesce(df, "fg_made", "legacy_field_goal_makes")
    out["fg_att"]    = _coalesce(df, "fg_att", "legacy_field_goal_attempts")
    out["pat_made"]  = _coalesce(df, "pat_made", "legacy_point_after_makes")

    # Fantasy
    out["fantasy_pts"]     = pd.to_numeric(df.get("fantasy_points"), errors="coerce")
    out["fantasy_pts_ppr"] = pd.to_numeric(df.get("fantasy_points_ppr"), errors="coerce")
    out["special_teams_tds"] = pd.to_numeric(df.get("special_teams_tds"), errors="coerce")

    # Drop rows with no player name
    out = out[out["player"].notna() & (out["player"].astype(str).str.strip() != "")]

    return out.reset_index(drop=True)


def aggregate_season(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the weekly table to season totals per player."""
    sum_cols = [
        "pass_cmp", "pass_att", "pass_yds", "pass_td", "pass_int",
        "rush_att", "rush_yds", "rush_td",
        "rec", "targets", "rec_yds", "rec_td",
        "def_sacks", "def_int", "def_tds",
        "fg_made", "fg_att", "pat_made",
        "fantasy_pts", "fantasy_pts_ppr", "special_teams_tds",
    ]
    mean_cols = ["pass_epa", "rush_epa", "rec_epa",
                 "target_share", "wopr", "racr", "pacr"]

    sum_cols  = [c for c in sum_cols if c in weekly.columns]
    mean_cols = [c for c in mean_cols if c in weekly.columns]

    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    agg["team"] = "last"
    agg["position"] = _mode_or_first
    agg["player"] = "first"      # keep a display name
    agg["week"] = "count"

    # Drop duplicate player-game rows before aggregating. The legacy source
    # sometimes carries more than one row for the same player and week, which
    # would otherwise inflate season totals and game counts. We key on
    # player_id so two different players sharing a name are NOT merged.
    key = "player_id" if "player_id" in weekly.columns else "player"
    weekly = weekly.drop_duplicates(subset=[key, "season", "week"]).copy()

    season = (
        weekly.groupby(["season", key], as_index=False)
        .agg(agg)
        .rename(columns={"week": "games_played"})
    )
    # If we grouped by player_id, 'player' is already in agg as the display name
    return season


def _mode_or_first(s: pd.Series):
    """Most common position for a player in a season (handles position changes)."""
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]


# Normalises raw position labels into clean groups.
# Historically fullbacks (FB) and halfbacks (HB) were the primary ball carriers,
# so they map to RB. Combo labels (e.g. "DE-DT", "G-T") take the first listed role.
POSITION_NORMALISE = {
    "HB": "RB", "FB": "RB", "TB": "RB", "WB": "RB",   # old backfield labels → RB
    "PK": "K",                                         # placekicker → K
    "OT": "T", "OG": "G",                              # offensive line variants
    "FS": "S", "SS": "S", "DB": "S",                   # defensive backs (non-CB) → S
    "ILB": "LB", "OLB": "LB", "MLB": "LB", "NT": "DT",
    "HB-FB": "RB", "FB-HB": "RB", "RB-FB": "RB", "FB-RB": "RB", "HB-RB": "RB",
}


def normalise_position(pos):
    """Map a raw position label to a clean canonical position."""
    if pd.isna(pos):
        return np.nan
    p = str(pos).strip().upper()
    if p in POSITION_NORMALISE:
        return POSITION_NORMALISE[p]
    # Combo positions like "DE-DT", "G-T" — take the first listed role, then remap
    if "-" in p:
        first = p.split("-")[0]
        return POSITION_NORMALISE.get(first, first)
    return p


def build_team_records(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Derive team win/loss records.

    The local file has game_won flags through ~2017 but not for recent
    seasons, so we fill 2018+ from the nflverse games schedule (which also
    gives us the team-abbreviation fix for relocated franchises).
    """
    # --- 1. Historical records from the local game_won flags ---
    games = weekly.dropna(subset=["game_won"]).copy()
    local_records = pd.DataFrame(columns=["season", "team", "wins", "losses", "win_pct"])

    if not games.empty:
        team_games = (
            games[["season", "team", "week", "game_won"]]
            .drop_duplicates(subset=["season", "team", "week"])
        )

        def _summ(grp):
            wins = (grp["game_won"] == True).sum()      # noqa: E712
            losses = (grp["game_won"] == False).sum()   # noqa: E712
            total = wins + losses
            return pd.Series({
                "wins": int(wins),
                "losses": int(losses),
                "win_pct": round(wins / total, 4) if total > 0 else np.nan,
            })

        local_records = (
            team_games.groupby(["season", "team"])
            .apply(_summ)
            .reset_index()
        )

    local_max_season = int(local_records["season"].max()) if not local_records.empty else 0

    # --- 2. Fill recent seasons (local_max+1 .. INGEST_END) from nflverse ---
    recent_records = _nflverse_team_records(local_max_season + 1, INGEST_END)

    combined = pd.concat([local_records, recent_records], ignore_index=True)
    combined = combined.drop_duplicates(subset=["season", "team"], keep="last")
    return combined.sort_values(["season", "team"]).reset_index(drop=True)


# Historical → current franchise abbreviation mapping (matches player stats)
_TEAM_ABBREV_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def _nflverse_team_records(start: int, end: int) -> pd.DataFrame:
    """Download nflverse games schedule and compute win% for [start, end]."""
    if start > end:
        return pd.DataFrame(columns=["season", "team", "wins", "losses", "win_pct"])

    import io
    import requests
    import pyarrow.parquet as pq

    url = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "schedules/games.parquet")
    try:
        print(f"  Fetching nflverse games for {start}–{end} team records...")
        resp = requests.get(url, headers={"User-Agent": "research python-requests"},
                            timeout=60)
        resp.raise_for_status()
        g = pq.read_table(io.BytesIO(resp.content)).to_pandas()
    except Exception as e:
        print(f"  WARN: could not fetch nflverse games ({e}); recent records skipped.")
        return pd.DataFrame(columns=["season", "team", "wins", "losses", "win_pct"])

    g = g[(g["game_type"] == "REG") & g["home_score"].notna() & g["away_score"].notna()]
    g = g[(g["season"] >= start) & (g["season"] <= end)]

    home = g[["season", "home_team", "result"]].rename(
        columns={"home_team": "team"})
    away = g[["season", "away_team", "result"]].rename(
        columns={"away_team": "team"})
    away["result"] = -away["result"]
    allg = pd.concat([home, away], ignore_index=True)

    rows = []
    for (season, team), grp in allg.groupby(["season", "team"]):
        wins = int((grp["result"] > 0).sum())
        losses = int((grp["result"] < 0).sum())
        ties = int((grp["result"] == 0).sum())
        total = wins + losses + ties
        team_norm = _TEAM_ABBREV_MAP.get(team, team)
        rows.append({
            "season": int(season),
            "team": team_norm,
            "wins": wins,
            "losses": losses,
            "win_pct": round((wins + 0.5 * ties) / total, 4) if total else np.nan,
        })
    return pd.DataFrame(rows)


def ingest():
    """Run the full local ingest and write the three raw CSVs."""
    df = load_raw()

    print("Building weekly table...")
    weekly = build_weekly(df)
    weekly.to_csv(RAW_DIR / "player_stats_weekly.csv", index=False)
    print(f"  Saved player_stats_weekly.csv ({len(weekly):,} rows, "
          f"{weekly['season'].min()}–{weekly['season'].max()})")

    print("Aggregating season totals...")
    season = aggregate_season(weekly)
    season.to_csv(RAW_DIR / "player_stats.csv", index=False)
    print(f"  Saved player_stats.csv ({len(season):,} rows)")

    print("Building team records...")
    teams = build_team_records(weekly)
    teams.to_csv(RAW_DIR / "team_records.csv", index=False)
    print(f"  Saved team_records.csv ({len(teams):,} team-seasons)")

    return season, weekly, teams


if __name__ == "__main__":
    ingest()
