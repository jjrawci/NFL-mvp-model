"""
scraper.py
----------
Scrapes the AP NFL MVP winners (1957–present) from Wikipedia and saves them to
data/raw/mvp_winners.csv. This is the label source for training.

Player stats and team records come from the local game dataset via
ingest_local.py — this module only handles the MVP winner list.
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project) python-requests/2.31"}

# Wikipedia AP MVP page
WIKI_MVP_URL = "https://en.wikipedia.org/wiki/Associated_Press_NFL_Most_Valuable_Player_Award"


# ---------------------------------------------------------------------------
# MVP winners — Wikipedia
# ---------------------------------------------------------------------------

def scrape_mvp_winners() -> pd.DataFrame:
    """
    Scrape the AP NFL MVP award table from Wikipedia.
    Returns a DataFrame with columns: season, player, position, team, mvp_winner.
    Covers 1957–present.
    """
    print("Scraping AP MVP winners from Wikipedia...")
    resp = requests.get(WIKI_MVP_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table", {"class": "wikitable"})

    # Find the year-by-year winners table
    mvp_table = None
    for t in tables:
        rows = t.find_all("tr")
        if rows:
            hdrs = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            if "Season" in hdrs and "Player" in hdrs:
                mvp_table = t
                break

    if mvp_table is None:
        raise RuntimeError("Could not find MVP table on Wikipedia page.")

    rows = mvp_table.find_all("tr")
    records = []
    for tr in rows[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            records.append(cells)

    # Handle variable row widths:
    #   Full row:  Season | Player | Position | Team | Votes | Ref
    #   Short row: Season | Player | Team | Votes | Ref  (position missing)
    KNOWN_POSITIONS = {
        "quarterback", "running back", "halfback", "fullback",
        "wide receiver", "tight end", "defensive end", "linebacker",
        "kicker", "safety", "cornerback", "QB", "RB", "WR", "TE",
        "DE", "LB", "K", "FB", "HB",
    }

    def _is_position(val):
        if not val or pd.isna(val):
            return False
        return any(p.lower() in val.lower() for p in KNOWN_POSITIONS)

    parsed = []
    for cells in records:
        if not cells or not cells[0] or len(cells) < 2:
            continue
        season_raw, player_raw = cells[0], cells[1]
        if len(cells) >= 4 and _is_position(cells[2]):
            position, team = cells[2], cells[3]
        elif len(cells) >= 3 and not _is_position(cells[2]):
            position, team = None, cells[2]
        else:
            position, team = None, (cells[2] if len(cells) > 2 else None)

        # The winner's vote share appears as "27 of 49 (55%)" somewhere in the
        # row. Capture the percentage as a 0–1 win margin signal.
        vote_share = None
        for cell in cells:
            m = re.search(r"\((\d+)\s*%\)", str(cell))
            if m:
                vote_share = int(m.group(1)) / 100.0
                break

        parsed.append({"season_raw": season_raw, "player_raw": player_raw,
                       "position": position, "team": team,
                       "winner_vote_share": vote_share})

    df = pd.DataFrame(parsed)

    # Forward-fill season (repeat-winner rows sometimes omit the year)
    df["season_raw"] = df["season_raw"].replace("", pd.NA).ffill()
    df["season"] = df["season_raw"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)

    # Strip ordinal suffixes like (2), (3) marking repeat winners
    df["player"] = df["player_raw"].str.replace(r"\(\d+\)", "", regex=True).str.strip()
    df["mvp_winner"] = 1
    df = df[["season", "player", "position", "team", "mvp_winner",
             "winner_vote_share"]].copy()

    # Split co-MVP years into separate rows
    CO_MVP_SPLITS = {
        "Brett FavreBarry Sanders": [("Brett Favre", "QB"), ("Barry Sanders", "RB")],
        "Peyton ManningSteve McNair": [("Peyton Manning", "QB"), ("Steve McNair", "QB")],
    }
    extra_rows, drop_idx = [], []
    for idx, row in df.iterrows():
        if row["player"] in CO_MVP_SPLITS:
            drop_idx.append(idx)
            for name, pos in CO_MVP_SPLITS[row["player"]]:
                extra_rows.append({"season": row["season"], "player": name,
                                   "position": pos, "team": None, "mvp_winner": 1,
                                   "winner_vote_share": row.get("winner_vote_share")})
    if drop_idx:
        df = df.drop(index=drop_idx)
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Normalise position labels (FB/HB grouped as RB, matching the ingest pipeline)
    pos_map = {
        "Quarterback": "QB", "Running back": "RB", "Halfback": "RB",
        "Fullback": "RB", "Wide receiver": "WR", "Tight end": "TE",
        "Defensive end": "DE", "Defensive tackle": "DT",
        "Linebacker": "LB", "Kicker": "K", "Placekicker": "K",
        "Safety": "S", "Cornerback": "CB", "QBRB": "QB",
    }
    for long, short in pos_map.items():
        df["position"] = df["position"].str.replace(long, short, case=False, regex=False)

    # Known positions for historical winners (fills gaps / fixes parser quirks)
    KNOWN_POS = {
        "Jim Brown": "RB", "Johnny Unitas": "QB", "Norm Van Brocklin": "QB",
        "Jim Taylor": "RB", "Y. A. Tittle": "QB", "Earl Morrall": "QB",
        "Roman Gabriel": "QB", "John Brodie": "QB", "O. J. Simpson": "RB",
        "Bert Jones": "QB", "Ken Anderson": "QB", "Dan Marino": "QB",
        "Boomer Esiason": "QB", "Joe Montana": "QB", "Steve Young": "QB",
        "Brett Favre": "QB", "Rich Gannon": "QB", "Peyton Manning": "QB",
        "LaDainian Tomlinson": "RB", "Tom Brady": "QB", "Cam Newton": "QB",
        "Matt Ryan": "QB", "Lamar Jackson": "QB", "Aaron Rodgers": "QB",
        "Patrick Mahomes": "QB", "Josh Allen": "QB", "Matthew Stafford": "QB",
        "Fran Tarkenton": "QB", "Terry Bradshaw": "QB", "Brian Sipe": "QB",
        "Ken Stabler": "QB", "Larry Brown": "RB", "Terrell Davis": "RB",
        "Kurt Warner": "QB", "Marshall Faulk": "RB", "Steve McNair": "QB",
        "Shaun Alexander": "RB", "Adrian Peterson": "RB", "Mark Moseley": "K",
        "Lawrence Taylor": "LB", "John Elway": "QB", "Thurman Thomas": "RB",
        "Emmitt Smith": "RB", "Barry Sanders": "RB", "Walter Payton": "RB",
        "Earl Campbell": "RB", "Marcus Allen": "RB", "Joe Theismann": "QB",
    }
    VALID_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "TE", "DT", "DE",
                       "LB", "K", "S", "CB", "DB"}
    df.loc[~df["position"].isin(VALID_POSITIONS), "position"] = pd.NA
    for player, pos in KNOWN_POS.items():
        df.loc[df["player"] == player, "position"] = pos

    df = df.sort_values("season").reset_index(drop=True)

    print(f"  Found {len(df)} MVP award seasons ({df['season'].min()}–{df['season'].max()})")
    df.to_csv(RAW_DIR / "mvp_winners.csv", index=False)
    print("  Saved mvp_winners.csv")
    return df


if __name__ == "__main__":
    scrape_mvp_winners()
