"""v3 stage 3: LightGBM on v2 features + stage-2 probability + transformer cross-encoder score.

  prep  (CPU)  rebuild v2 features and honest (out-of-fold) stage-1/stage-2 probabilities from the
               saved v2 fold models - no retraining - and write the pair lists the cross-encoder
               should score (plausible pairs only: prob2 >= CE_MIN_PROB and rank <= CE_MAX_RANK).
  fit   (CPU)  join cross-encoder scores, train stage 3 two-fold on a subset of training entities,
               tune the decision rule on the validation entities (global / per-source thresholds,
               expected-F0.5), score test, write output/*.tsv.
The cross-encoder itself is trained and applied by cross_encoder.py on a GPU machine.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import Config
from data import load_gt, load_prepared
from run_pipeline import load_train_cands
from evaluate import f05_entity
from matching import to_lists, write_lists
from model import (PROB_FEATURES, STAGE2_FEATURES, assign_best, decide_expected_f, decide_threshold,
                   feature_matrix, predict_two, prob_features, two_fold)

CE_MIN_PROB = 0.01
CE_MAX_RANK = 8
SUP_K = 6
SUP_COLS = ["sup_name_max", "sup_addr_max", "sup_prob_of_best", "sup_n_similar", "sup_weighted",
            "sup_other_src_name_max"]
STAGE3_EXTRA = ["prob2"] + [f"p2_{f}" for f in PROB_FEATURES[1:]] + \
               ["ce", "ce_scored", "q_rank_ce", "q_gap_ce", "p_rank_ce", "p_gap_ce", "ce_minus_p2"] + SUP_COLS
STAGE3_FEATURES = STAGE2_FEATURES + STAGE3_EXTRA


def ce_candidates(meta: pd.DataFrame) -> np.ndarray:
    """Rows worth a cross-encoder score - identical rule for train and test."""
    rank = meta.groupby("q")["prob2"].rank(ascending=False, method="first").to_numpy()
    return (meta["prob2"].to_numpy() >= CE_MIN_PROB) & (rank <= CE_MAX_RANK)


def _support_chunk(args):
    """For each candidate among an entity's top-K (by prob2): similarity to the entity's OTHER top-K
    candidates. Records of one real business (from S2 and S3) resemble each other; a look-alike
    neighbour usually does not resemble the rest of the cluster."""
    from rapidfuzz import fuzz
    groups = args
    out = []
    for rows, names, addrs, probs, srcs in groups:
        n = len(rows)
        for i in range(n):
            bn = ba = bp = bo = 0.0
            nsim = 0
            wsum = 0.0
            for j in range(n):
                if i == j:
                    continue
                sn = fuzz.token_set_ratio(names[i], names[j]) / 100.0
                sa = fuzz.token_set_ratio(addrs[i], addrs[j]) / 100.0 if addrs[i] and addrs[j] else 0.0
                if sn > bn:
                    bn, bp = sn, probs[j]
                ba = max(ba, sa)
                if srcs[i] != srcs[j]:
                    bo = max(bo, sn)
                if sn >= 0.8 or sa >= 0.9:
                    nsim += 1
                wsum += probs[j] * max(sn, sa)
            out.append((rows[i], bn, ba, bp, nsim, wsum, bo))
    return out


def support_features(meta: pd.DataFrame, pool: pd.DataFrame, n_jobs: int) -> np.ndarray:
    t = time.time()
    res = np.full((len(meta), len(SUP_COLS)), np.nan, dtype=np.float32)
    rank = meta.groupby("q")["prob2"].rank(ascending=False, method="first").to_numpy()
    top = np.flatnonzero(rank <= SUP_K)
    top = top[np.argsort(meta["q"].to_numpy()[top], kind="stable")]
    qs = meta["q"].to_numpy()[top]
    bounds = np.flatnonzero(np.diff(qs)) + 1
    names, addrs = pool["name_key"].to_numpy(), pool["addr_norm"].to_numpy()
    pv, pr, sr = meta["p"].to_numpy(), meta["prob2"].to_numpy(), meta["src"].to_numpy()
    groups = []
    for g in np.split(top, bounds):
        if len(g) > 1:
            p = pv[g]
            groups.append((g.tolist(), names[p].tolist(), addrs[p].tolist(), pr[g].tolist(), sr[g].tolist()))
    chunks = [groups[i:i + 20_000] for i in range(0, len(groups), 20_000)]
    if n_jobs > 1 and len(chunks) > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as pl:
            parts = pl.map(_support_chunk, chunks)
    else:
        parts = [_support_chunk(c) for c in chunks]
    for part in parts:
        if part:
            arr = np.array(part, dtype=np.float64)
            res[arr[:, 0].astype(np.int64)] = arr[:, 1:].astype(np.float32)
    print(f"[s3-support] {len(groups):,} entities, {len(top):,} rows in {time.time() - t:.0f}s", flush=True)
    return res


def _load_models(cfg, stage):
    return [lgb.Booster(model_file=cfg.path(f"model_{stage}_{i}.txt")) for i in (0, 1)]


def cmd_prep(cfg, a):
    t = time.time()
    ms1, ms2 = _load_models(cfg, "s1"), _load_models(cfg, "s2")
    if not a.test_only:            # a synthetic world (syn_world.py) has only a test split
        # ---------------- train: reproduce the exact split / folds of run_pipeline.stage_train
        s1, pool = load_prepared(cfg, "train")
        c, u = load_train_cands(cfg, s1)
        c = c[(u < cfg.val_s1_frac + cfg.train_s1_frac)[c["q"].values]].reset_index(drop=True)
        qv, pv = c["q"].values, c["p"].values
        is_val = u[qv] < cfg.val_s1_frac
        fold = (u[qv] >= cfg.val_s1_frac + cfg.train_s1_frac / 2).astype(np.int8)
        X = feature_matrix(c, s1, pool, cfg.n_jobs)

        def oof(models, X):
            prob = np.zeros(len(X), dtype=np.float32)
            for f in (0, 1):
                other = (fold != f) & ~is_val
                prob[other] = models[f].predict(X[other], num_threads=cfg.n_jobs)
                prob[is_val] += 0.5 * models[f].predict(X[is_val], num_threads=cfg.n_jobs)
            return prob

        prob1 = oof(ms1, X)
        X = np.hstack([X, prob_features(qv, pv, pool["src"].values[pv], prob1)])
        prob2 = oof(ms2, X)
        np.save(cfg.path("s3_train_X.npy"), X)
        meta = pd.DataFrame({"q": qv, "p": pv, "is_val": is_val, "fold": fold, "prob1": prob1, "prob2": prob2,
                             "src": pool["src"].values[pv]})
        # stage-3 entities: all validation entities + a sample of the other training entities
        rng = np.random.default_rng(7)
        tr_q = np.unique(qv[~is_val])
        keep_q = set(np.unique(qv[is_val]).tolist()) | set(rng.choice(tr_q, size=min(a.stage3_train_s1, len(tr_q)), replace=False).tolist())
        meta["in_s3"] = meta["q"].isin(keep_q)
        meta["need_ce"] = ce_candidates(meta) & meta["in_s3"]
        sub = meta["in_s3"].to_numpy()
        sup = np.full((len(meta), len(SUP_COLS)), np.nan, dtype=np.float32)
        sup[sub] = support_features(meta[sub].reset_index(drop=True), pool, cfg.n_jobs)
        for i, col in enumerate(SUP_COLS):
            meta[col] = sup[:, i]
        meta.to_parquet(cfg.path("s3_train_meta.parquet"), index=False)
        meta.loc[meta["need_ce"], ["q", "p"]].to_parquet(cfg.path("s3_train_ce_pairs.parquet"), index=False)
        print(f"[s3-prep] train {len(meta):,} pairs, stage-3 entities {len(keep_q):,}, "
              f"CE pairs {int(meta['need_ce'].sum()):,}; {time.time() - t:.0f}s", flush=True)
        del X

    # ---------------- test
    s1, pool = load_prepared(cfg, "test")
    c = pd.read_parquet(cfg.path("test_cands.parquet"))
    qv, pv = c["q"].values, c["p"].values
    X = feature_matrix(c, s1, pool, cfg.n_jobs)
    prob1 = predict_two(ms1, X, cfg.n_jobs)
    X = np.hstack([X, prob_features(qv, pv, pool["src"].values[pv], prob1)])
    prob2 = predict_two(ms2, X, cfg.n_jobs)
    np.save(cfg.path("s3_test_X.npy"), X)
    meta = pd.DataFrame({"q": qv, "p": pv, "prob1": prob1, "prob2": prob2, "src": pool["src"].values[pv]})
    meta["need_ce"] = ce_candidates(meta)
    sup = support_features(meta, pool, cfg.n_jobs)
    for i, col in enumerate(SUP_COLS):
        meta[col] = sup[:, i]
    meta.to_parquet(cfg.path("s3_test_meta.parquet"), index=False)
    meta.loc[meta["need_ce"], ["q", "p"]].to_parquet(cfg.path("s3_test_ce_pairs.parquet"), index=False)
    print(f"[s3-prep] test {len(meta):,} pairs, CE pairs {int(meta['need_ce'].sum()):,}; {time.time() - t:.0f}s", flush=True)


def _stage3_matrix(X, meta, ce_path):
    ce = pd.read_parquet(ce_path)
    m = meta[["q", "p"]].merge(ce, on=["q", "p"], how="left")
    meta = meta.assign(ce=m["ce"].to_numpy(np.float32))
    scored = meta["ce"].notna()
    g = meta[scored].groupby("q")["ce"]
    gp = meta[scored].groupby("p")["ce"]
    extra = pd.DataFrame(index=meta.index, dtype=np.float32)
    extra["prob2"] = meta["prob2"].to_numpy(np.float32)
    P2 = prob_features(meta["q"].values, meta["p"].values, meta["src"].values, meta["prob2"].values)[:, 1:]
    for i, f in enumerate(PROB_FEATURES[1:]):
        extra[f"p2_{f}"] = P2[:, i]
    extra["ce"] = meta["ce"]
    extra["ce_scored"] = scored.astype(np.float32)
    extra["q_rank_ce"] = np.nan; extra["q_gap_ce"] = np.nan; extra["p_rank_ce"] = np.nan; extra["p_gap_ce"] = np.nan
    extra.loc[scored, "q_rank_ce"] = g.rank(ascending=False, method="min")
    extra.loc[scored, "q_gap_ce"] = g.transform("max") - meta.loc[scored, "ce"]
    extra.loc[scored, "p_rank_ce"] = gp.rank(ascending=False, method="min")
    extra.loc[scored, "p_gap_ce"] = gp.transform("max") - meta.loc[scored, "ce"]
    extra["ce_minus_p2"] = extra["ce"] - extra["prob2"]
    for col in SUP_COLS:
        extra[col] = meta[col].to_numpy(np.float32)
    return np.hstack([X, extra[STAGE3_EXTRA].to_numpy(np.float32)])


def decide_threshold_src(best, t2, t3):
    thr = np.where(best["src"].to_numpy() == 2, t2, t3)
    return best[best["prob"].to_numpy() >= thr][["q", "p", "prob"]]


def _eval(sel, true_qp, val_q):
    pred = {}
    for q, p in zip(sel["q"].values, sel["p"].values):
        pred.setdefault(q, set()).add(p)
    return float(np.mean([f05_entity(pred.get(q, set()), true_qp.get(q, set())) for q in val_q]))


def cmd_fit(cfg, a):
    t = time.time()
    s1, pool = load_prepared(cfg, "train")
    meta = pd.read_parquet(cfg.path("s3_train_meta.parquet"))
    X = np.load(cfg.path("s3_train_X.npy"), mmap_mode="r")
    sub = meta["in_s3"].to_numpy()
    X3 = _stage3_matrix(np.asarray(X[sub]), meta[sub].reset_index(drop=True), cfg.path("train_ce.parquet"))
    m3 = meta[sub].reset_index(drop=True)
    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index)]
    gq, gp = s1_pos[gt["source1_entity_id"]].values, p_pos[gt["entity_id"]].values
    y = np.isin(m3["q"].values.astype(np.int64) * len(pool) + m3["p"].values,
                gq.astype(np.int64) * len(pool) + gp).astype(np.int8)
    params = dict(cfg.lgb_params, num_threads=cfg.n_jobs, seed=cfg.seed)
    ms3, prob3 = two_fold(X3, y, m3["fold"].values, m3["is_val"].values, params, cfg.lgb_rounds,
                          STAGE3_FEATURES, cfg.n_jobs)
    for i, m in enumerate(ms3):
        m.save_model(cfg.path(f"model_s3_{i}.txt"))

    # validation: competition across all train pairs, stage-3 prob where available else stage-2
    prob_all = meta["prob2"].to_numpy().copy()
    prob_all[np.flatnonzero(sub)] = prob3
    val_q = np.unique(meta.loc[meta["is_val"], "q"].values)
    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)                 # includes entities without candidates
    vset = set(val_q.tolist())
    true_qp = {}
    for x, z in zip(gq, gp):
        if x in vset:
            true_qp.setdefault(x, set()).add(z)
    res = {}
    for name, prob in (("stage2", meta["prob2"].to_numpy()), ("stage3", prob_all)):
        best = assign_best(meta["q"].values, meta["p"].values, prob)
        best = best[best["q"].isin(vset)].merge(meta[["q", "p", "src"]], on=["q", "p"])
        for thr in np.round(np.arange(0.4, 0.951, 0.025), 3):
            res[(name, "thr", float(thr), float(thr))] = _eval(decide_threshold(best, thr), true_qp, val_q)
        for alpha in (0.0, 0.1, 0.2):
            res[(name, "expf", alpha, alpha)] = _eval(decide_expected_f(best, alpha), true_qp, val_q)
        t0 = max((k for k in res if k[0] == name and k[1] == "thr"), key=lambda k: res[k])[2]
        t2, t3 = t0, t0
        grid = np.round(np.arange(0.4, 0.951, 0.025), 3)
        for _ in range(2):
            t2 = max(grid, key=lambda x: _eval(decide_threshold_src(best, x, t3), true_qp, val_q))
            t3 = max(grid, key=lambda x: _eval(decide_threshold_src(best, t2, x), true_qp, val_q))
        res[(name, "src", float(t2), float(t3))] = _eval(decide_threshold_src(best, t2, t3), true_qp, val_q)
    key = max(res, key=res.get)
    for k in sorted(res, key=lambda k: -res[k])[:8]:
        print(f"[s3-fit] {k}: val macro F0.5 {res[k]:.5f}", flush=True)
    rep = {"decision": {"stage": key[0], "rule": key[1], "p1": key[2], "p2": key[3]}, "val_macro_f05": res[key],
           "best_stage2": max(v for k, v in res.items() if k[0] == "stage2"),
           "all": {"|".join(map(str, k)): v for k, v in res.items()}}
    json.dump(rep, open(cfg.path("stage3_report.json"), "w"), indent=1)
    print(f"[s3-fit] chosen {key}: {res[key]:.5f} (best stage-2 rule {rep['best_stage2']:.5f}); {time.time() - t:.0f}s", flush=True)
    del X, X3

    # ---------------- test
    s1, pool = load_prepared(cfg, "test")
    meta = pd.read_parquet(cfg.path("s3_test_meta.parquet"))
    if key[0] == "stage3":
        X = np.load(cfg.path("s3_test_X.npy"), mmap_mode="r")
        prob = np.empty(len(meta), dtype=np.float32)
        step = 10_000_000
        X3 = _stage3_matrix(np.asarray(X), meta, cfg.path("test_ce.parquet"))
        for s in range(0, len(meta), step):
            prob[s:s + step] = predict_two(ms3, X3[s:s + step], cfg.n_jobs)
    else:
        prob = meta["prob2"].to_numpy()
    best = assign_best(meta["q"].values, meta["p"].values, prob).merge(meta[["q", "p", "src"]], on=["q", "p"])
    if key[1] == "thr":
        sel = decide_threshold(best, key[2])
    elif key[1] == "expf":
        sel = decide_expected_f(best, key[2])
    else:
        sel = decide_threshold_src(best, key[2], key[3])
    s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), s1_ids, to_lists(meta, s1_ids, pool_ids), "candidate_entity_ids")
    matches = to_lists(sel, s1_ids, pool_ids)
    write_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), s1_ids, matches, "matched_entity_ids")
    print(f"[s3-predict] {len(s1_ids):,} S1, {len(matches):,} with matches, {len(sel):,} matched ids; {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "fit"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--stage3-train-s1", type=int, default=500_000)
    ap.add_argument("--drop-s1-frac", type=float, default=Config.drop_s1_frac)
    ap.add_argument("--train-s1-frac", type=float, default=Config.train_s1_frac)
    ap.add_argument("--test-only", action="store_true", help="prep only the test split")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, out_dir=a.out_dir, n_jobs=a.n_jobs, drop_s1_frac=a.drop_s1_frac, train_s1_frac=a.train_s1_frac)
    cmd_prep(cfg, a) if a.cmd == "prep" else cmd_fit(cfg, a)


if __name__ == "__main__":
    main()
