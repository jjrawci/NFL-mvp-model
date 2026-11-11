# NFL MVP Prediction (v3)

Predicts NFL AP MVP odds from season stats. A blend of a learning-to-rank model
and a logistic regression, trained on every modern MVP race (1999–2025), scores
a pool of players for a given season and outputs each player's probability of
winning MVP.

## Results

Leave-one-season-out cross-validation over 27 seasons (1999–2025), averaged over
8 random seeds of the ranker (the logistic part is deterministic):

| Model | Top-1 | Top-3 | Top-5 | MRR |
|-------|-------|-------|-------|-----|
| Ranker alone (v2) | 64% | 88% | 94% | 0.768 |
| Logistic alone | 70% | 89% | 100% | 0.814 |
| **Blend (v3, used)** | 73% | 86% | 98% | 0.808 |

MRR is the mean reciprocal rank of the true winner (1.0 = always ranked first).

How much to trust this: with 27 seasons these differences are small relative to
the noise. The blend beat the ranker alone by +0.041 ± 0.020 MRR, but was level
with logistic alone (-0.006 ± 0.036). The logistic label ("received any votes")
and its regularization were picked after looking at the same seasons, so the
true edge is probably smaller than the table suggests. The blend was chosen over
logistic alone because its weight was fixed in advance and it depends less on one
tuned setting, not because it scored higher. A single ranker seed can swing
top-1 from about 56% to 74%.

Top-3 and top-5 are the most reliable measures of quality. With only ~27 MVP
races, top-1 is noisy because the closest two-player races (e.g. Stafford vs.
Maye 2025, decided by one first-place vote) are near coin-flips the stats can't
separate.

## How it works

**Learning to rank.** MVP voting is a per-season ranking problem, not binary
classification. The ranker (XGBoost's ranking objective) is trained on the full
AP voting data as graded relevance: a near-unanimous winner is a 4, a player who
got a stray vote is a 1, non-vote-getters are 0. It learns to order players
within each season.

**Blended with a logistic regression.** A regularized logistic regression is
trained to predict whether a player received any MVP votes. Each model's scores
are standardized across the season's offensive pool and averaged 50/50. The two
models make correlated but different mistakes (per-season correlation about
0.73), so the blend is more stable than either alone.

**Odds.** Scores are turned into percentages with a softmax over the top 15
contenders. The softmax scale (2.0) is fitted on out-of-fold scores from past
seasons, so stated odds track how often favourites actually win. The earlier
v2 odds were underconfident: the favourite averaged 45% but won 63% of the time.

**One model over all offensive candidates.** QBs, RBs, WRs and TEs are ranked
together so the model learns their relative order directly (e.g. a historic RB
season vs. a strong QB season). Defenders and kickers are out of scope — no
non-offensive player has won since 1986.

**Prediction matches training.** Many features are relative to whoever is in the
pool (league z-scores, ranks, the "full season" cutoff). Training only kept
players with some recorded production, so prediction applies the same filter
(`has_production()` in `src/features.py`) and includes the late-season features.
An earlier version of v3 skipped both: the prediction pool was padded with
linemen and other zero-stat players, which shifted the median games played and
every league-relative feature, and the late-season features were set to 0. Only
60 of the 100 model features matched the training data. They now all match, and
the accuracy cross-validation was unaffected because it used the training data
directly. The visible change was in current-season odds (for 2026 the favourite
moved from about 45% to 49%), with little change in the ordering.

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

### Data: what's in the repo and what isn't

Included in the repo, so a clone can predict right away:
- the trained model (`data/processed/mvp_model_modern.pkl` and `_meta.json`)
- AP MVP winners (`data/raw/mvp_winners.csv`)
- the exported voting tables (`data/raw/mvp_votes/mvp_votes_<year>.xlsx`, 1999–2025)

Not included (too large or regenerable, listed in `.gitignore`):
- **`NFL_All_Player_Game_Data_1950_2026.parquet`**: the game-by-game player
  stats. This is the one input you must supply yourself. It is too big for GitHub
  and goes in the project root with that exact filename. It needs nflverse-style
  columns (`season`, `week`, `player_id`, `season_type`, `game_won`, box-score
  stats), as read by `src/ingest_local.py`.
- Files generated from it by `python run.py ingest` and `python run.py features`:
  `data/raw/player_stats.csv`, `player_stats_weekly.csv`, `team_records.csv` and
  `data/processed/features.csv`.
- Downloaded caches: `data/raw/nflverse_weekly_<season>.parquet` (from
  `python run.py refresh`) and `data/raw/nflverse_games.parquet` (the schedule,
  fetched on demand).

To predict from a fresh clone, add the parquet, run `python run.py ingest`, then
`python run.py predict --season 2026`. To retrain, also run
`python run.py features` and then `python run.py train`. `refresh` and the
schedule download need internet access to GitHub (nflverse releases).

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
python run.py refresh    # pull the latest weeks of the current season, rebuild features
python run.py mvp        # scrape AP MVP winners from Wikipedia
python run.py features   # build the engineered feature dataset
python run.py train      # train the ranker + logistic blend
```

## Project layout

```
MVPproject/
├── run.py                    # CLI entry point
├── src/
│   ├── ingest_local.py       # local parquet → raw CSVs
│   ├── scraper.py            # AP MVP winners (Wikipedia)
│   ├── votes.py              # parse full PFR voting tables (data/raw/mvp_votes/)
│   ├── voter_bias.py         # voter-bias features (tested, not used by the model)
│   ├── features.py           # feature engineering (shared by train + predict)
│   ├── features_spec.py      # feature sets + position-track definitions
│   ├── model_ranker.py       # ranker + logistic blend (train + save)
│   ├── schedule.py           # schedule features (tested, not used)
│   ├── electorate.py         # vote simulation for odds (tested, not used)
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

## Voter bias

Voters are not random. They favor quarterbacks (50 of the 71 MVPs since 1957),
winning teams and strong finishes. The model trains on real vote results, so it
already absorbs the biases that show up in the stats.

We also tested whether extra, systematic biases add anything. `src/voter_bias.py`
builds ten features from information known before each season: reputation
(prior-year vote share), voter fatigue (past wins, defending MVP), comeback
stories (games missed last year) and team turnarounds (change in win %). Two tests:

- **Added to the ranker**, compared over 8 random seeds: changes in mean
  reciprocal rank of -0.007 to +0.008, all within one standard error of zero.
- **Two-stage re-ranker** (the ranker's score plus a few bias features, applied to
  each year's top 8, with fully nested leave-one-season-out validation): worse
  than the ranker alone for every feature set, with seasons improving and
  worsening about equally.

No measurable effect was detected, so these features are not in the model. That
is not proof the biases don't exist. With only 27 seasons the confidence
intervals are wide (a defending MVP's vote share came out +2.5 points, 95%
interval -1.8 to +6.9), so small effects can't be ruled out. Worth re-testing as
more seasons accumulate.

Because top-1 accuracy shifts by about ±7 points with the random seed alone,
feature comparisons here use several seeds and mean reciprocal rank rather than
a single-seed top-1.

## Other experiments that didn't help

All compared over 8 seeds against the baseline (change in mean reciprocal rank,
± standard error):

| Idea | Change in MRR |
|------|---------------|
| Strength of schedule (`src/schedule.py`) | +0.002 ± 0.011 |
| Primetime exposure | +0.005 ± 0.011 |
| Cast discount (production minus team volume) | -0.008 ± 0.005 |
| Margin over the next-best player | -0.045 ± 0.006 (clearly worse) |
| Interaction features (win % x production) | -0.009 ± 0.013 |
| All-time percentile comparisons | +0.000 ± 0.013 |
| Median instead of zero imputation | -0.003 ± 0.010 |

Only the margin features were clearly harmful; the rest are within noise, so
they were left out to keep the model simple.

**Electorate simulation** (`src/electorate.py`). Simulates 50 voters with a shared
per-player "narrative" shock plus individual noise, 10-5-3-2-1 points, and counts
wins. Held-out log loss was 1.131, versus 1.126 for a softmax with its scale
re-fitted. That is a tie, so the simpler softmax is used. Both beat the old
untuned scale (1.244). The simulation's voter-noise setting also landed on the
edge of the search grid, so it is not well pinned down.

## Versions

- **v1** (tagged `v1`): logistic-regression models, binary winner target, two
  eras (modern + legacy 1957–1998). ~61% top-1 modern.
- **v2** (tagged `v2`): learning-to-rank on full voting data, modern-only
  (1999–2025), tuned. Better top-3/top-5, cleaner codebase.
- **v3** (current): ranker + logistic blend, odds scale fitted on past seasons.
  The schedule, voter-bias and other feature experiments above were tested and
  left out.

Check out the `v1` or `v2` tags to see the earlier approaches.
