"""
votes.py
--------
Loads manually-exported Pro Football Reference MVP voting tables.

PFR blocks automated scraping, so you export each year's voting table by hand:
  1. Go to e.g. https://www.pro-football-reference.com/awards/awards_2023.htm
  2. On the "AP MVP Voting" table, click Share & Export → "Get table as CSV"
  3. Paste into a file named  data/raw/mvp_votes/mvp_votes_<year>.csv

This module reads whatever year files are present and returns a tidy frame of
(season, player, vote_pts, first_place_votes, vote_share) that the feature
pipeline can merge in as a graded training target.

It is tolerant of PFR's exact column names and the junk header rows their CSV
export sometimes includes.
"""

import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

VOTES_DIR = Path(__file__).parent.parent / "data" / "raw" / "mvp_votes"


def _clean_name(name) -> str:
    if pd.isna(name):
        return ""
    # strip PFR markers and vote annotations
    return re.sub(r"[*+]", "", str(name)).strip()


def _find_col(cols, *candidates):
    """Return the first column whose lowercased name contains any candidate."""
    low = {c: str(c).lower().strip() for c in cols}
    for cand in candidates:
        for c, l in low.items():
            if cand in l:
                return c
    return None


def _parse_one(path: Path) -> pd.DataFrame:
    """Parse one PFR MVP-voting CSV export into tidy rows."""
    # Extract the season from the filename (mvp_votes_2023.csv -> 2023)
    m = re.search(r"(\d{4})", path.name)
    if not m:
        return pd.DataFrame()
    season = int(m.group(1))

    # PFR exports carry a category super-header (Games/Passing/Rushing/...) on
    # the first row, with the real column names (Rk,Pos,Player,...) on the
    # second. Depending on the export, pandas may read the whole thing into one
    # comma-joined column. We handle both: try normal headers first, and if the
    # frame collapsed to one column, split it manually.
    is_excel = path.suffix.lower() in (".xlsx", ".xls")

    def _read(skip):
        if is_excel:
            return pd.read_excel(path, skiprows=skip, header=None, dtype=str)
        return pd.read_csv(path, skiprows=skip, header=None, dtype=str)

    raw = None
    try:
        rows = _read(0)
        # If everything landed in one column, split each cell on commas
        if rows.shape[1] == 1:
            split = rows.iloc[:, 0].astype(str).str.split(",", expand=True)
            # Find the row that is the real header (contains 'Player')
            hdr_idx = None
            for i in range(min(3, len(split))):
                if split.iloc[i].astype(str).str.contains("Player", case=False, na=False).any():
                    hdr_idx = i
                    break
            if hdr_idx is None:
                print(f"  WARN: no header row found in {path.name}")
                return pd.DataFrame()
            header = split.iloc[hdr_idx].tolist()
            body = split.iloc[hdr_idx + 1:].copy()
            body.columns = header
            raw = body
        else:
            # Multi-column read: the real header may be row 0 or row 1
            for hdr in (0, 1):
                tmp = _read(hdr)
                tmp.columns = tmp.iloc[0]
                cand = tmp.iloc[1:]
                if _find_col(cand.columns, "player"):
                    raw = cand
                    break
    except Exception as e:
        print(f"  WARN: error reading {path.name}: {e}")
        return pd.DataFrame()

    if raw is None or _find_col(raw.columns, "player") is None:
        print(f"  WARN: could not parse {path.name}")
        return pd.DataFrame()

    player_col = _find_col(raw.columns, "player")
    pts_col    = _find_col(raw.columns, "vote pts", "pts won", "points", "pts")
    first_col  = _find_col(raw.columns, "1st place", "first", "1st")
    share_col  = _find_col(raw.columns, "share")

    out = pd.DataFrame()
    out["player"] = raw[player_col].apply(_clean_name)
    out = out[out["player"] != ""]
    out = out[~out["player"].str.lower().isin(["player", "league average"])]

    out["vote_pts"] = pd.to_numeric(raw[pts_col], errors="coerce") if pts_col else np.nan
    out["first_place_votes"] = pd.to_numeric(raw[first_col], errors="coerce") if first_col else np.nan

    if share_col:
        share = raw[share_col].astype(str).str.replace("%", "", regex=False)
        share = pd.to_numeric(share, errors="coerce")
        # PFR share is 0–1 already in most exports; if it looks like a percent, scale
        if share.max(skipna=True) is not np.nan and share.max() > 1.5:
            share = share / 100.0
        out["vote_share"] = share
    else:
        out["vote_share"] = np.nan

    out["season"] = season
    # If share is missing but we have points, derive share within the year
    if out["vote_share"].isna().all() and out["vote_pts"].notna().any():
        total = out["vote_pts"].sum()
        if total > 0:
            out["vote_share"] = (out["vote_pts"] / total).round(4)

    return out[["season", "player", "vote_pts", "first_place_votes", "vote_share"]]


def load_all_votes() -> pd.DataFrame:
    """Load and combine every exported voting file present (.csv or .xlsx)."""
    files = sorted(glob.glob(str(VOTES_DIR / "mvp_votes_*.csv")) +
                   glob.glob(str(VOTES_DIR / "mvp_votes_*.xlsx")))
    if not files:
        return pd.DataFrame(columns=["season", "player", "vote_pts",
                                     "first_place_votes", "vote_share"])
    frames = [_parse_one(Path(f)) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["season", "player", "vote_pts",
                                     "first_place_votes", "vote_share"])
    combined = pd.concat(frames, ignore_index=True)
    combined["player"] = combined["player"].apply(_clean_name)
    return combined


def available_seasons() -> list:
    """Which seasons have an exported voting file."""
    df = load_all_votes()
    return sorted(df["season"].unique().tolist()) if not df.empty else []


if __name__ == "__main__":
    df = load_all_votes()
    if df.empty:
        print(f"No voting files found in {VOTES_DIR}")
        print("Export PFR awards tables as mvp_votes_<year>.csv to enable vote-share targets.")
    else:
        print(f"Loaded voting data for seasons: {available_seasons()}")
        print(df.groupby("season").size().to_string())
