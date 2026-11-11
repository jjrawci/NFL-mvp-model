"""
model_ranker.py
---------------
Production MVP model for the modern era (1999+), using learning-to-rank.

Why a ranker: MVP voting is fundamentally a per-season ranking problem, not a
binary classification. Training on the full exported vote data as graded
relevance labels (near-unanimous winner = 4 ... got a stray vote = 1 ... no
votes = 0) and optimising an NDCG ranking objective within each season beats
the binary logistic model on top-1 (69% vs 65%) and top-3 (88% vs 81%), and it
directly learns the ordering (e.g. that a 98%-vote winner belongs above a
gaudy-stat runner-up).

A single ranker scores all offensive players (QB/RB/WR/TE) together, since
ranking is comparative. In v3 it is blended 50/50 with a logistic regression
trained to predict "received any MVP votes" (see ensemble_cv). Defensive and
kicker candidates are out of scope.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

# feature sets + position-track assignment
import features_spec as spec

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Train through 2025 (last completed season). 2026 is in progress — no MVP
# awarded yet — so it's prediction-only, not a training season.
START, END = 1999, 2025
# The ranker works best as ONE model over all offensive candidates (QB+RB+WR+TE)
# ranked together within each season — ranking is inherently comparative, so we
# want Peterson and Brady in the same group to learn their relative order.
OFFENSE_TRACKS = ("qb", "skill")

XGB_PARAMS = dict(
    objective="rank:ndcg",
    n_estimators=200,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=2,
    random_state=42,
    n_jobs=2,
)


# Ensemble: the ranker is blended 50/50 with a logistic regression trained to
# predict "received any MVP votes". The two make correlated but not identical
# mistakes; in 8-seed leave-one-season-out CV the blend beat the ranker alone
# (MRR ~0.81 vs ~0.77). The blend weight was fixed in advance, not tuned.
LOGIT_C = 0.01
W_RANKER = 0.5
ODDS_POOL = 15
ODDS_SCALES = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
PROB_FLOOR = 1e-3


def vote_relevance(vs: float) -> int:
    """Bucket a vote share into an integer relevance grade for the ranker."""
    if vs >= 0.5:
        return 4
    if vs >= 0.2:
        return 3
    if vs >= 0.05:
        return 2
    if vs > 0:
        return 1
    return 0


def _offense_features(df: pd.DataFrame) -> list:
    """Union of qb + skill feature sets (ranker scores all offense together)."""
    feats = []
    for track in OFFENSE_TRACKS:
        for f in spec.get_feature_cols(df, track):
            if f in df.columns and f not in feats:
                feats.append(f)
    return feats


def _fit_ranker(sub: pd.DataFrame, feats: list, params: dict = None):
    """
    Fit an XGBRanker on one track. Rows must be sorted by season so the group
    sizes line up. Returns (ranker, scaler).
    """
    sub = sub.sort_values("season").reset_index(drop=True)
    X = sub[feats].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = sub["rel"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    groups = [int((sub["season"] == s).sum()) for s in sorted(sub["season"].unique())]
    r = XGBRanker(**(params or XGB_PARAMS))
    r.fit(Xs, y, group=groups)
    return r, scaler


def _clean_X(df: pd.DataFrame, feats: list) -> np.ndarray:
    return df[feats].replace([np.inf, -np.inf], np.nan).fillna(0).values


def _fit_logit(sub: pd.DataFrame, feats: list):
    """Logistic regression for 'received any MVP votes'. Returns (model, scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(_clean_X(sub, feats))
    y = (sub["vote_share"].fillna(0) > 0).astype(int).values
    m = LogisticRegression(C=LOGIT_C, max_iter=3000)
    m.fit(X, y)
    return m, scaler


def _zscore(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def blend_scores(ranker_scores, logit_scores, w_ranker: float = W_RANKER) -> np.ndarray:
    """Blend two models' scores after standardising each within the candidate pool."""
    return w_ranker * _zscore(ranker_scores) + (1 - w_ranker) * _zscore(logit_scores)


def _winner_ranks(off: pd.DataFrame, pred: np.ndarray) -> dict:
    o = off.assign(pred=pred)
    out = {}
    for s, g in o.groupby("season"):
        if g["mvp_winner"].sum() == 0:
            continue
        r = g["pred"].rank(ascending=False, method="min")
        out[int(s)] = int(r[g["mvp_winner"] == 1].min())
    return out


def _summarize(ranks: dict) -> dict:
    rk = np.array(list(ranks.values()), dtype=float)
    out = {f"top_{n}_acc": round(float((rk <= n).mean()), 3) for n in (1, 2, 3, 5)}
    out["mrr"] = round(float((1.0 / rk).mean()), 4)
    out["mean_rank"] = round(float(rk.mean()), 3)
    return out


def ensemble_cv(off: pd.DataFrame, feats: list, seed: int = None) -> dict:
    """
    Leave-one-season-out CV of the ranker, the logistic model, and their blend.
    Returns metrics for each plus the blended out-of-fold scores.
    """
    off = off.reset_index(drop=True)
    rcv = cross_validate(off, feats, seed=seed)
    lcv = np.zeros(len(off))
    for s in sorted(off["season"].unique()):
        tr = off[off["season"] != s]
        te_idx = off.index[off["season"] == s]
        m, sc = _fit_logit(tr, feats)
        lcv[te_idx] = m.decision_function(sc.transform(_clean_X(off.loc[te_idx], feats)))

    blend = np.zeros(len(off))
    for s in sorted(off["season"].unique()):
        idx = off.index[off["season"] == s].values
        blend[idx] = blend_scores(rcv["oof_pred"][idx], lcv[idx])

    return {
        "ranker": _summarize(rcv["winner_ranks"]),
        "logistic": _summarize(_winner_ranks(off, lcv)),
        "ensemble": _summarize(_winner_ranks(off, blend)),
        "blend_oof": blend,
    }


def fit_odds_scale(off: pd.DataFrame, blend_oof: np.ndarray) -> float:
    """
    Choose the softmax scale that turns blended scores into well-calibrated win
    probabilities: minimise the log loss of the actual winner over past seasons,
    using the top-15 contender pool and out-of-fold scores.
    """
    off = off.reset_index(drop=True).assign(pred=blend_oof)
    pools = []
    for s, g in off.groupby("season"):
        g = g.sort_values("pred", ascending=False).head(ODDS_POOL)
        w = g["mvp_winner"].values.astype(float)
        if w.sum() == 0:
            continue
        pools.append((_zscore(g["pred"].values), w / w.sum()))
    best, best_loss = None, np.inf
    for sc in ODDS_SCALES:
        loss = 0.0
        for z, w in pools:
            p = np.exp(z * sc)
            p /= p.sum()
            loss += -(w * np.log(np.clip(p, PROB_FLOOR, 1))).sum()
        if loss < best_loss:
            best, best_loss = sc, loss
    return float(best)


def cross_validate(off: pd.DataFrame, feats: list, seed: int = None) -> dict:
    """
    Leave-one-season-out CV: one ranker over ALL offensive players per fold.

    Returns top-N accuracy (was the actual MVP in the model's top N?) plus two
    smoother measures that don't swing on a single coin-flip season:
      mrr       : mean reciprocal rank of the winner (1.0 = always #1)
      mean_rank : average rank the model gave the actual winner
    and `winner_ranks`, a {season: rank} dict for diagnosing individual years.
    """
    off = off.reset_index(drop=True)
    params = dict(XGB_PARAMS)
    if seed is not None:
        params["random_state"] = seed
    cv = np.zeros(len(off))
    for s in sorted(off["season"].unique()):
        tr = off[off["season"] != s]
        te = off[off["season"] == s]
        if tr["mvp_winner"].sum() == 0 or te.empty:
            continue
        r, scaler = _fit_ranker(tr, feats, params)
        Xte = scaler.transform(te[feats].replace([np.inf, -np.inf], np.nan).fillna(0).values)
        cv[te.index.values] = r.predict(Xte)
    off = off.assign(pred=cv)

    winner_ranks = {}
    for s, g in off.groupby("season"):
        if g["mvp_winner"].sum() == 0:
            continue
        ranks = g["pred"].rank(ascending=False, method="min")
        winner_ranks[int(s)] = int(ranks[g["mvp_winner"] == 1].min())  # best-ranked co-winner

    rk = np.array(list(winner_ranks.values()), dtype=float)
    out = {f"top_{n}_acc": round(float((rk <= n).mean()), 3) for n in (1, 2, 3, 5)}
    out["mrr"] = round(float((1.0 / rk).mean()), 4)
    out["mean_rank"] = round(float(rk.mean()), 3)
    out["winner_ranks"] = winner_ranks
    out["oof_pred"] = cv          # out-of-fold score for every row (aligned to `off`)
    return out


def _load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError("features.csv not found — run `python run.py features` first.")
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["season"] >= START) & (df["season"] <= END)].copy().reset_index(drop=True)
    df["_track"] = df["position"].apply(spec.assign_track)
    if "vote_share" not in df.columns:
        df["vote_share"] = df["mvp_winner"].astype(float)
    df["rel"] = df["vote_share"].fillna(0).apply(vote_relevance)
    return df


def train() -> dict:
    """Train, cross-validate (leave-one-season-out), and save the ranker model."""
    df = _load_features()
    off = df[df["_track"].isin(OFFENSE_TRACKS)].copy().reset_index(drop=True)
    print(f"[ranker] {len(off)} offensive player-seasons ({START}-{END}), "
          f"{int((off['mvp_winner']==1).sum())} MVP winners, "
          f"{int((off['vote_share']>0).sum())} vote-getters.")

    feats = _offense_features(off)
    cvres = ensemble_cv(off, feats, seed=XGB_PARAMS["random_state"])

    print(f"\n=== Leave-one-season-out CV (modern {START}-{END}, ranker seed "
          f"{XGB_PARAMS['random_state']}) ===")
    for name in ("ranker", "logistic", "ensemble"):
        m = cvres[name]
        print(f"  {name:9s} top-1 {m['top_1_acc']:.1%}  top-3 {m['top_3_acc']:.1%}  "
              f"top-5 {m['top_5_acc']:.1%}  mrr {m['mrr']:.3f}  "
              f"mean rank {m['mean_rank']:.2f}")
    print("  (single-seed top-1 swings about +/-7 pts; see README for 8-seed figures)")

    odds_scale = fit_odds_scale(off, cvres["blend_oof"])
    print(f"  odds softmax scale fitted on out-of-fold blended scores: {odds_scale}")

    # Fit final models on all offensive data
    print("\nFitting final ranker and logistic model on full dataset...")
    r, scaler = _fit_ranker(off, feats)
    lm, lscaler = _fit_logit(off, feats)
    bundle = {"kind": "ensemble", "era": "modern",
              "tracks": {"offense": {"ranker": r, "scaler": scaler,
                                     "logit": lm, "logit_scaler": lscaler,
                                     "w_ranker": W_RANKER, "odds_scale": odds_scale,
                                     "feature_cols": feats, "base_rate": 1.0}}}
    with open(PROCESSED_DIR / "mvp_model_modern.pkl", "wb") as f:
        pickle.dump(bundle, f)

    top_n = {k: v for k, v in cvres["ensemble"].items() if k.startswith("top_")}
    meta = {
        "era": "modern", "model_type": "ranker+logistic ensemble",
        "start_season": START, "end_season": END,
        "n_features": len(feats), "w_ranker": W_RANKER, "logit_c": LOGIT_C,
        "odds_scale": odds_scale,
        "cv_ranker": cvres["ranker"], "cv_logistic": cvres["logistic"],
        "cv_ensemble": cvres["ensemble"], **top_n,
    }
    with open(PROCESSED_DIR / "mvp_model_modern_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved ranker to {PROCESSED_DIR / 'mvp_model_modern.pkl'}")
    return {"model": bundle, "metrics": meta}


if __name__ == "__main__":
    train()
