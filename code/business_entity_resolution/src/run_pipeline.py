"""End-to-end pipeline: data -> normalisation -> blocking -> features -> LightGBM -> matches.

Stages (run all, or pick with --stages):
  translit  learn non-Latin -> English token table from training pairs
  prepare   normalise train/test records -> work/*.parquet
  block     candidate generation + context features -> work/{split}_cands.parquet
  train     two-stage LightGBM (2-fold stacking), decision rule tuned on held-out S1 entities
  predict   score test candidates, write output/matching_results.tsv + candidate_pairs.tsv
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import block, blocking_recall, rescue_empty_address
from config import Config
from data import learn_translit, load_gt, load_prepared, prepare
from evaluate import f05_entity
from features import FEATURES, context_features
from matching import to_lists, write_lists
from model import (STAGE2_FEATURES, assign_best, decide_expected_f, decide_threshold, feature_matrix,
                   predict_two, prob_features, two_fold)


def stage_block(cfg, split):
    s1, pool = load_prepared(cfg, split)
    if cfg.block_countries:
        # re-block only some countries (e.g. after a country-specific normalisation change) and
        # merge into the existing candidate table; context features are recomputed for all rows
        only = [x.strip().lower() for x in cfg.block_countries.split(",")]
        old = pd.read_parquet(cfg.path(f"{split}_cands.parquet"), columns=["q", "p", "name_cos", "addr_cos", "comb"])
        old = old[~np.isin(s1["country"].values[old["q"].values], only)]
        c = pd.concat([old, block(s1, pool, cfg, countries=set(only))], ignore_index=True)
        print(f"[block:{split}] re-blocked {only}, kept {len(old):,} existing pairs", flush=True)
    else:
        c = block(s1, pool, cfg)
    c = add_rescue(c, s1, pool, cfg)
    c = context_features(c)
    c.to_parquet(cfg.path(f"{split}_cands.parquet"), index=False)
    msg = f"[block:{split}] {len(c):,} pairs, {len(c) / len(s1):.1f} per S1, reduction ratio {1 - len(c) / (len(s1) * len(pool)):.6f}"
    if split == "train":
        rec, n = blocking_recall(c, s1, pool, load_gt(cfg))
        msg += f", recall ceiling {rec:.4f} over {n:,} true pairs"
    print(msg, flush=True)


def add_rescue(c, s1, pool, cfg):
    """Union the empty-address rescue pairs into a candidate table (existing pairs win)."""
    if cfg.rescue_k <= 0:
        return c
    base = c[["q", "p", "name_cos", "addr_cos", "comb"]]
    extra = rescue_empty_address(s1, pool, cfg)
    key = lambda d: d["q"].values.astype(np.int64) * len(pool) + d["p"].values
    extra = extra[~np.isin(key(extra), key(base))]
    print(f"[rescue] +{len(extra):,} new pairs ({len(extra) / len(s1):.1f} per S1)", flush=True)
    return pd.concat([base, extra], ignore_index=True)


def stage_rescue(cfg, split):
    """Add rescue pairs to an existing candidate table and recompute its context features."""
    s1, pool = load_prepared(cfg, split)
    c = pd.read_parquet(cfg.path(f"{split}_cands.parquet"), columns=["q", "p", "name_cos", "addr_cos", "comb"])
    c = context_features(add_rescue(c, s1, pool, cfg))
    c.to_parquet(cfg.path(f"{split}_cands.parquet"), index=False)
    msg = f"[rescue:{split}] {len(c):,} pairs, {len(c) / len(s1):.1f} per S1"
    if split == "train":
        rec, n = blocking_recall(c, s1, pool, load_gt(cfg))
        msg += f", recall ceiling {rec:.4f} over {n:,} true pairs"
    print(msg, flush=True)


def _eval(sel: pd.DataFrame, true_qp: dict, val_q, country=None):
    """Macro F0.5 (+ mean precision/recall over entities with predictions / true matches)."""
    pred = {}
    for q, p in zip(sel["q"].values, sel["p"].values):
        pred.setdefault(q, set()).add(p)
    f, ps, rs = [], [], []
    for q in val_q:
        P, T = pred.get(q, set()), true_qp.get(q, set())
        f.append(f05_entity(P, T))
        if P:
            ps.append(len(P & T) / len(P))
        if T:
            rs.append(len(P & T) / len(T))
    return float(np.mean(f)), float(np.mean(ps)) if ps else 0.0, float(np.mean(rs)) if rs else 0.0


def load_train_cands(cfg, s1):
    """Training candidates with the test-like distractor density: entities with
    u >= 1 - drop_s1_frac are removed (disjoint from the train/val entities) and the
    competition features are recomputed without them."""
    u = np.random.default_rng(cfg.seed).random(len(s1))
    c = pd.read_parquet(cfg.path("train_cands.parquet"))
    if cfg.drop_s1_frac > 0:
        assert cfg.val_s1_frac + cfg.train_s1_frac <= 1 - cfg.drop_s1_frac
        c = c[u[c["q"].values] < 1 - cfg.drop_s1_frac][["q", "p", "name_cos", "addr_cos", "comb"]]
        c = context_features(c.reset_index(drop=True))
        print(f"[train] distractor simulation: dropped {cfg.drop_s1_frac:.0%} of S1 entities, {len(c):,} pairs left", flush=True)
    return c, u


def stage_train(cfg):
    t = time.time()
    s1, pool = load_prepared(cfg, "train")
    c, u = load_train_cands(cfg, s1)
    use_q = u < cfg.val_s1_frac + cfg.train_s1_frac
    c = c[use_q[c["q"].values]].reset_index(drop=True)
    qv, pv = c["q"].values, c["p"].values
    is_val = u[qv] < cfg.val_s1_frac
    fold = (u[qv] >= cfg.val_s1_frac + cfg.train_s1_frac / 2).astype(np.int8)

    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index) & gt["source1_entity_id"].isin(s1_pos.index)]
    gq, gp = s1_pos[gt["source1_entity_id"]].values, p_pos[gt["entity_id"]].values
    y = np.isin(qv.astype(np.int64) * len(pool) + pv, gq.astype(np.int64) * len(pool) + gp).astype(np.int8)

    X = feature_matrix(c, s1, pool, cfg.n_jobs)
    print(f"[train] {len(c):,} pairs ({y.mean():.3f} positive); val pairs {is_val.sum():,}; "
          f"features {X.shape[1]}; {time.time() - t:.0f}s", flush=True)
    params = dict(cfg.lgb_params, num_threads=cfg.n_jobs, seed=cfg.seed)
    ms1, prob1 = two_fold(X, y, fold, is_val, params, cfg.lgb_rounds, FEATURES, cfg.n_jobs)
    src = pool["src"].values[pv]
    X = np.hstack([X, prob_features(qv, pv, src, prob1)])
    ms2, prob2 = two_fold(X, y, fold, is_val, params, cfg.lgb_rounds, STAGE2_FEATURES, cfg.n_jobs)
    del X
    for i, m in enumerate(ms1):
        m.save_model(cfg.path(f"model_s1_{i}.txt"))
    for i, m in enumerate(ms2):
        m.save_model(cfg.path(f"model_s2_{i}.txt"))

    # ---- decision tuning on validation entities; competition resolved across ALL scored pairs
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    true_qp = {}
    for a, b in zip(gq, gp):
        if a in vset:
            true_qp.setdefault(a, set()).add(b)
    results = {}
    for name, prob in (("stage1", prob1), ("stage2", prob2)):
        best = assign_best(qv, pv, prob)
        best = best[best["q"].isin(vset)]
        for thr in np.round(np.arange(0.3, 0.951, 0.05), 3):
            results[(name, "thr", float(thr))] = _eval(decide_threshold(best, thr), true_qp, val_q)
        for alpha in (0.0, 0.05, 0.1, 0.2, 0.3):
            results[(name, "expf", alpha)] = _eval(decide_expected_f(best, alpha), true_qp, val_q)
    key = max(results, key=lambda k: results[k][0])
    for k in sorted(results, key=lambda k: -results[k][0])[:8]:
        print(f"[train] {k}: F0.5 {results[k][0]:.4f}  P {results[k][1]:.4f}  R {results[k][2]:.4f}", flush=True)

    # per-country breakdown for the chosen rule
    prob = prob2 if key[0] == "stage2" else prob1
    best = assign_best(qv, pv, prob)
    best = best[best["q"].isin(vset)]
    sel = decide_threshold(best, key[2]) if key[1] == "thr" else decide_expected_f(best, key[2])
    by_country = {}
    for ctry in np.unique(s1["country"].values[val_q]):
        vq = val_q[s1["country"].values[val_q] == ctry]
        by_country[ctry] = _eval(sel[sel["q"].isin(set(vq.tolist()))], true_qp, vq)
    imp = sorted(zip(STAGE2_FEATURES, ms2[0].feature_importance("gain")), key=lambda x: -x[1])
    report = {"decision": {"stage": key[0], "rule": key[1], "param": key[2]},
              "val_macro_f05": results[key][0], "val_precision": results[key][1], "val_recall": results[key][2],
              "by_country": {k: {"f05": v[0], "precision": v[1], "recall": v[2]} for k, v in by_country.items()},
              "val_entities": int(len(val_q)),
              "all_results": {f"{a}|{b}|{c_}": v for (a, b, c_), v in results.items()},
              "importance_top25": [(f, float(v)) for f, v in imp[:25]]}
    with open(cfg.path("train_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[train] chosen {key}: val macro F0.5 {results[key][0]:.4f}; by country "
          f"{ {k: round(v[0], 4) for k, v in by_country.items()} }; {time.time() - t:.0f}s", flush=True)


def stage_predict(cfg):
    t = time.time()
    s1, pool = load_prepared(cfg, "test")
    c = pd.read_parquet(cfg.path("test_cands.parquet"))
    rep = json.load(open(cfg.path("train_report.json")))["decision"]
    ms1 = [lgb.Booster(model_file=cfg.path(f"model_s1_{i}.txt")) for i in (0, 1)]
    ms2 = [lgb.Booster(model_file=cfg.path(f"model_s2_{i}.txt")) for i in (0, 1)]
    qv, pv = c["q"].values, c["p"].values
    X = feature_matrix(c, s1, pool, cfg.n_jobs)
    prob = predict_two(ms1, X, cfg.n_jobs)
    if rep["stage"] == "stage2":
        X = np.hstack([X, prob_features(qv, pv, pool["src"].values[pv], prob)])
        prob = predict_two(ms2, X, cfg.n_jobs)
    del X
    scored = pd.DataFrame({"q": qv, "p": pv, "prob": prob})
    scored.to_parquet(cfg.path("test_scored.parquet"), index=False)
    best = assign_best(qv, pv, prob)
    sel = decide_threshold(best, rep["param"]) if rep["rule"] == "thr" else decide_expected_f(best, rep["param"])

    s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), s1_ids, to_lists(c, s1_ids, pool_ids), "candidate_entity_ids")
    matches = to_lists(sel, s1_ids, pool_ids)
    write_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), s1_ids, matches, "matched_entity_ids")
    by_c = pd.Series(s1["country"].values).groupby(s1["country"].values).size()
    with_m = pd.Series([s1["country"].values[q] for q in np.unique(sel["q"].values)]).value_counts()
    print(f"[predict] {len(s1_ids):,} S1 entities, {len(matches):,} with matches, {len(sel):,} matched ids "
          f"({rep}); share with matches by country {(with_m / by_c).round(3).to_dict()}; {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--stages", default="translit,prepare,block,train,predict")
    ap.add_argument("--train-s1-frac", type=float, default=Config.train_s1_frac)
    ap.add_argument("--val-s1-frac", type=float, default=Config.val_s1_frac)
    ap.add_argument("--joint-w", type=float, default=Config.joint_w)
    ap.add_argument("--k-joint", type=int, default=Config.k_joint)
    ap.add_argument("--k-name-extra", type=int, default=Config.k_name_extra)
    ap.add_argument("--name-max-df", type=float, default=Config.name_max_df)
    ap.add_argument("--block-countries", default="", help="re-block only these countries (comma list) and merge")
    ap.add_argument("--rescue-k", type=int, default=Config.rescue_k)
    ap.add_argument("--drop-s1-frac", type=float, default=Config.drop_s1_frac)
    ap.add_argument("--splits", default="train,test", help="splits for the prepare/block stages")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, out_dir=a.out_dir, n_jobs=a.n_jobs,
                 train_s1_frac=a.train_s1_frac, val_s1_frac=a.val_s1_frac, joint_w=a.joint_w,
                 k_joint=a.k_joint, k_name_extra=a.k_name_extra, name_max_df=a.name_max_df,
                 block_countries=a.block_countries, rescue_k=a.rescue_k, drop_s1_frac=a.drop_s1_frac)
    os.makedirs(cfg.work_dir, exist_ok=True)
    stages = a.stages.split(",")
    t = time.time()
    if "translit" in stages:
        learn_translit(cfg)
    splits = a.splits.split(",")
    if "prepare" in stages:
        for sp in splits:
            prepare(cfg, sp)
    if "block" in stages:
        for sp in splits:
            stage_block(cfg, sp)
    if "rescue" in stages:
        for sp in splits:
            stage_rescue(cfg, sp)
    if "train" in stages:
        stage_train(cfg)
    if "predict" in stages:
        stage_predict(cfg)
    print(f"[done] total {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
