# NFL MVP Prediction

Predicts NFL AP MVP odds from season stats. Trained on every AP MVP season since
1957, the model scores a pool of players for a given season and outputs each
player's probability of winning MVP.

## How it works

Two models, one per era, because the game — and the stats available to describe
it — changed over time:

| Model  | Seasons   | Features                                  | Top-1 | Top-3 |
|--------|-----------|-------------------------------------------|-------|-------|
| modern | 1999–2024 | full set incl. EPA & advanced metrics     | 65.4% | 76.9% |
| legacy | 1957–1998 | traditional box-score stats (no EPA)      | 52.5% | 77.5% |

"Top-1" = how often the model's #1 pick was the actual MVP (leave-one-season-out
cross-validation). "Top-3" = how often the actual MVP was in the model's top 3.

### Position tracks

MVP voting rewards different things by position — a QB essentially must be on an
elite team, while a running back can win by carrying a heavier individual load on
a weaker team. A single model can't hold both patterns, so each model trains
separate per-position tracks and combines them with dampened historical base
rates (so a QB and an RB are compared fairly):

- **modern** splits into `qb`, `skill` (RB/WR/TE), and `defkick` (defense/kickers)
- **legacy** merges `qb`+`skill` into one `offense` track (its smaller data does
  better merged) plus a `defkick` track

### What the model looks at

- **Production**: passing/rushing/receiving yards & TDs, efficiency (passer
  rating, EPA, YPC), per-game rates
- **League context**: z-scores and ranks vs. other players at the same kind of
  position that season
- **Team success**: win %, whether the team was elite/playoff-caliber, win% vs.
  the league average
- **Carrying vs. being carried**: the player's share of their team's total
  offense and touchdowns, and the strength of their supporting cast
- **Late-season form**: production over the final 6 weeks vs. earlier (voters
  remember strong finishes)
- **Dominance & milestones**: historically freakish seasons (e.g. 2,000 rushing
  yards, 5,000 passing yards)

## Setup

```bash
pip install -r requirements.txt
```

The historical game data lives in `NFL_All_Player_Game_Data_1950_2026.parquet`
in the project root.

## Usage

### Predict a season
```bash
python run.py predict --season 2024 --era modern      # recent seasons
python run.py predict --season 1984 --era legacy       # pre-1999 seasons
```

### Predict from your own CSV
Provide a pool of players with their season box-score stats (see
`data/current_season_template.csv` for the columns):
```bash
python run.py predict --input data/current_season_template.csv --era modern
```
Include all the realistic contenders in the pool — the model's comparison
features (ranks, carry share) are computed across whoever you provide.

### Rebuild everything from scratch
```bash
python run.py pipeline        # ingest → mvp → features → train (both eras)
```

Or step by step:
```bash
python run.py ingest          # process the local parquet into raw CSVs
python run.py mvp             # scrape AP MVP winners from Wikipedia
python run.py features        # build the engineered feature dataset
python run.py train --era all # train modern + legacy models
```

## Project layout

```
MVPproject/
├── run.py                       # CLI entry point
├── src/
│   ├── ingest_local.py          # local parquet → raw season/weekly/team CSVs
│   ├── scraper.py               # AP MVP winners from Wikipedia
│   ├── features.py              # feature engineering (shared by train + predict)
│   ├── model.py                 # per-era, per-track training & evaluation
│   └── predict.py               # scoring a player pool
├── data/
│   ├── raw/                     # player_stats, player_stats_weekly, team_records,
│   │                            #   mvp_winners (generated)
│   ├── processed/               # features.csv + trained models + feature weights
│   └── current_season_template.csv
└── NFL_All_Player_Game_Data_1950_2026.parquet
```

## Known limitations

- **Non-offensive MVPs** (Alan Page 1971 DT, Mark Moseley 1982 K, Lawrence Taylor
  1986 LB) are near-impossible to predict — there are only 3 in 68 years, too few
  for the model to learn the pattern. The `defkick` track keeps them in the pool
  but ranks them poorly. No defender has won since 1986.
- **Close two-player races** (e.g. Rodgers vs. Mahomes 2020/21, Lamar vs. Allen
  2023/24) often come down to voter narrative the stats can't capture. These show
  up as the model's top-2 being right but top-1 being a coin flip.
- **Pre-1999 data** lacks EPA and advanced metrics, so the legacy model relies on
  traditional stats and team record — still effective, just less precise.

## Why logistic regression (not XGBoost)?

With one MVP per season there are only ~27 positive training examples in the
modern era and ~69 across all history. That's far too few for gradient boosting
to learn reliable non-linear splits — in testing, XGBoost (classifier and ranker)
topped out around 42% top-1 vs. 61.5% for logistic regression. Linear models
generalize better when positive examples are scarce.
