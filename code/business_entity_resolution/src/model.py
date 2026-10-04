"""Two-stage LightGBM matcher with out-of-fold stacking and expected-F0.5 decisions.

Stage 1  pair features -> P(match)
Stage 2  pair features + "competition" features built from stage-1 probabilities
         (how this pair ranks among the S1 entity's candidates and among all S1 entities
          claiming the same S2/S3 record)
Both stages are trained 2-fold over S1 entities so every training pair gets an honest
out-of-fold probability; test uses the average of the two fold models.

Decision: each S2/S3 record goes to its most probable S1 entity (true for 100% of training
pairs); then per S1 entity we keep the top-j candidates maximising expected F0.5.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, build

PROB_FEATURES = ["prob", "q_rank_prob", "q_gap_prob", "q_sum_prob", "q_n_strong", "q_second_prob",
                 "p_rank_prob", "p_gap_prob", "p_n_strong", "p_second_prob", "q_src_rank_prob"]
STAGE2_FEATURES = FEATURES + PROB_FEATURES


def feature_matrix(c: pd.DataFrame, s1, pool, n_jobs, step=5_000_000) -> np.ndarray:
    parts = []
    for s in range(0, len(c), step):
        parts.append(build(c.iloc[s:s + step], s1, pool, n_jobs)[FEATURES].to_numpy(np.float32))
    return np.vstack(parts) if parts else np.zeros((0, len(FEATURES)), np.float32)


def _second(keys, vals):
    """Second-largest value per group (0 if the group has one element), aligned to rows."""
    df = pd.DataFrame({"k": keys, "v": vals})
    srt = df.sort_values(["k", "v"], ascending=[True, False])
    sec = srt[srt.groupby("k").cumcount() == 1].set_index("k")["v"]
    return df["k"].map(sec).fillna(0).to_numpy(np.float32)


def prob_features(q, p, src, prob) -> np.ndarray:
    df = pd.DataFrame({"q": q, "p": p, "src": src, "prob": prob})
    gq, gp = df.groupby("q")["prob"], df.groupby("p")["prob"]
    strong = (df["prob"] > 0.5).astype(np.int16)
    out = np.column_stack([
        df["prob"].to_numpy(np.float32),
        gq.rank(ascending=False, method="min").to_numpy(np.float32),
        (gq.transform("max") - df["prob"]).to_numpy(np.float32),
        gq.transform("sum").to_numpy(np.float32),
        strong.groupby(df["q"]).transform("sum").to_numpy(np.float32),
        _second(df["q"].to_numpy(), df["prob"].to_numpy()),
        gp.rank(ascending=False, method="min").to_numpy(np.float32),
        (gp.transform("max") - df["prob"]).to_numpy(np.float32),
        strong.groupby(df["p"]).transform("sum").to_numpy(np.float32),
        _second(df["p"].to_numpy(), df["prob"].to_numpy()),
        df.groupby(["q", "src"])["prob"].rank(ascending=False, method="min").to_numpy(np.float32),
    ])
    return out


def fit(X, y, rows, val_rows, params, rounds, names):
    t = time.time()
    dtr = lgb.Dataset(X[rows], y[rows], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[val_rows], y[val_rows], reference=dtr)
    m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(250)])
    print(f"[model] fit {rows.sum():,} rows -> best iter {m.best_iteration}, "
          f"val logloss {m.best_score['valid_0']['binary_logloss']:.5f} in {time.time() - t:.0f}s", flush=True)
    return m


def two_fold(X, y, fold, is_val, params, rounds, names, n_jobs):
    """fold: 0/1 per row for non-validation rows. Returns (models, out-of-fold probs for all rows)."""
    ms, prob = [], np.zeros(len(y), dtype=np.float32)
    for f in (0, 1):
        m = fit(X, y, (fold == f) & ~is_val, is_val, params, rounds, names)
        other = (fold != f) & ~is_val
        prob[other] = m.predict(X[other], num_threads=n_jobs)
        prob[is_val] += 0.5 * m.predict(X[is_val], num_threads=n_jobs)
        ms.append(m)
    return ms, prob


def predict_two(ms, X, n_jobs):
    return np.mean([m.predict(X, num_threads=n_jobs) for m in ms], axis=0).astype(np.float32)


# ---------------------------------------------------------------- decisions
def assign_best(q, p, prob):
    """Keep, for every S2/S3 record p, only its highest-probability S1 entity."""
    df = pd.DataFrame({"q": q, "p": p, "prob": prob})
    return df.sort_values("prob", ascending=False).drop_duplicates("p")


def decide_threshold(best: pd.DataFrame, thr: float) -> pd.DataFrame:
    return best[best["prob"] >= thr][["q", "p", "prob"]]


def decide_expected_f(best: pd.DataFrame, alpha: float = 0.0, floor: float = 0.05) -> pd.DataFrame:
    """Per S1 entity, keep the top-j candidates maximising plug-in expected F0.5.
    E[#true] = sum of candidate probs + alpha (mass of matches blocking may have missed)."""
    d = best[best["prob"] >= floor].sort_values(["q", "prob"], ascending=[True, False]).copy()
    g = d.groupby("q")["prob"]
    d["j"] = g.cumcount() + 1
    d["tp"] = g.cumsum()
    d["T"] = g.transform("sum") + alpha
    d["f"] = 1.25 * d["tp"] / (1.25 * d["tp"] + 0.25 * (d["T"] - d["tp"]) + (d["j"] - d["tp"]))
    # F of predicting nothing = P(no true match) ~ prod(1 - p_i) (and 0 mass missed)
    f0 = np.exp(np.log1p(-d["prob"].clip(upper=0.999999)).groupby(d["q"]).sum())
    if alpha > 0:
        f0 = f0 * np.exp(-alpha)
    best_j = d.loc[d.groupby("q")["f"].idxmax(), ["q", "j", "f"]].set_index("q")
    keep_q = best_j[best_j["f"] > f0.reindex(best_j.index).values]
    d = d[d["q"].isin(keep_q.index)]
    d = d[d["j"] <= d["q"].map(keep_q["j"])]
    return d[["q", "p", "prob"]]
