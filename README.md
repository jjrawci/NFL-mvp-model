# NFL MVP Prediction (v2)

Predicts NFL AP MVP odds from season stats. A learning-to-rank model trained on
every modern MVP race (1999–2025) scores a pool of players for a given season
and outputs each player's probability of winning MVP.

## Results

Leave-one-season-out cross-validation over 27 seasons (1999–2025):

| Metric | Score | Meaning |
|--------|-------|---------|
| Top-1  | 63%   | model's #1 pick was the actual MVP |
| Top-3  | 89%   | actual MVP was in the model's top 3 |
| Top-5  | 93%   | actual MVP was in the model's top 5 |

Top-3 and top-5 are the most reliable measures of quality — with only ~27 MVP
races, top-1 is noisy because the closest two-player races (e.g. Stafford vs.
Maye 2025, decided by one first-place vote) are near coin-flips the stats can't
separate.

## How it works

**Learning to rank.** MVP voting is a per-season ranking problem, not binary
classification. The model (XGBoost's ranking objective) is trained on the full
AP voting data as graded relevance: a near-unanimous winner is a 4, a player who
got a stray vote is a 1, non-vote-getters are 0. It learns to order players
within each season, which beats a binary winner/not-winner classifier.

**One model over all offensive candidates.** QBs, RBs, WRs and TEs are ranked
together so the model learns their relative order directly (e.g. a historic RB
season vs. a strong QB season). Defenders and kickers are out of scope — no
non-offensive player has won since 1986.

### Features

- **Production & efficiency** — passing/rushing/receiving yards & TDs, passer
  rating, EPA, per-game rates
- **League-relative comparisons** — z-scores and ranks vs. same-position peers
  that season; a "led the league in passing yards" flag
- **Team success** — win %, elite/playoff-team flags, win% vs. league average
- **Carrying vs. being carried** — share of the team's total offense and TDs,
  and supporting-cast strength
- **Late-season form** — production over the final 6 weeks vs. earlier
- **Dominance & milestones** — historically great seasons (2,000 rush yards, etc.)

## Setup

```bash
pip install -r requirements.txt
```

The historical game data lives in `NFL_All_Player_Game_Data_1950_2026.parquet`
in the project root.

## Usage

```bash
# Predict a season (builds the candidate pool automatically)
python run.py predict --season 2026        # current season odds
python run.py predict --season 2025         # a completed season

# Predict from your own CSV of player box-score stats
python run.py predict --input data/current_season_template.csv

# Rebuild everything from scratch
python run.py pipeline                      # ingest → mvp → features → train
```

Individual steps:
```bash
python run.py ingest     # local parquet → raw season/weekly/team CSVs
python run.py mvp        # scrape AP MVP winners from Wikipedia
python run.py features   # build the engineered feature dataset
python run.py train      # train the ranker
```

## Project layout

```
MVPproject/
├── run.py                    # CLI entry point
├── src/
│   ├── ingest_local.py       # local parquet → raw CSVs
│   ├── scraper.py            # AP MVP winners (Wikipedia)
│   ├── votes.py              # parse full PFR voting tables (data/raw/mvp_votes/)
│   ├── features.py           # feature engineering (shared by train + predict)
│   ├── features_spec.py      # feature sets + position-track definitions
│   ├── model_ranker.py       # the learning-to-rank model (train + save)
│   └── predict.py            # score a player pool
├── data/
│   ├── raw/                  # generated stats + MVP winners + voting tables
│   ├── processed/            # features.csv + trained model
│   └── current_season_template.csv
└── NFL_All_Player_Game_Data_1950_2026.parquet
```

## Known limitations

- **Close two-player races** (Stafford/Maye 2025, Rodgers/Mahomes 2020–21,
  Lamar/Allen 2023–24) often come down to voter narrative the stats can't see.
  These show up as top-2 being right but top-1 a coin flip.
- **Anomalous winners** like Adrian Peterson 2012 (2,097 rushing yards but only
  13 TDs on a mediocre team, winning largely on an ACL-comeback narrative) are
  near-impossible to predict from a box score.
- **Defensive/kicker MVPs** (3 ever, none since 1986) are out of scope.

## v1 vs v2

- **v1** (tagged `v1`): logistic-regression models, binary winner target, two
  eras (modern + legacy 1957–1998). ~61% top-1 modern.
- **v2** (current): learning-to-rank on full voting data, modern-only
  (1999–2025), tuned. Better top-3/top-5, cleaner codebase.

Check out the `v1` tag to see the earlier approach.
