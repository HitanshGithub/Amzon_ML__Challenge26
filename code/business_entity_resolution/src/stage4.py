"""v4 = v3 + France self-training + wider cross-encoder coverage + stage-3 seed ensemble.

  diag   where does v3 lose score: cross-encoder coverage of true pairs, validation P/R per
         country and per loss reason, per-country score distributions on test
  pairs  write the v4 pair lists (reusing v3 candidates and features):
           s4_train_ce_pairs / s4_test_ce_pairs   pairs the cross-encoder must score
           s4_ce_train_pairs                      cross-encoder training pairs with stage-2
                                                  hard negatives (train split, labelled)
           s4_pseudo_<country>                    pseudo-labelled test pairs for countries
                                                  absent from training (France): pairs where
                                                  v3's LightGBM and cross-encoder agree strongly
  fit    stage 3 with all train entities, pseudo-labelled rows, several seeds; decision rule
         tuned on the validation entities; test probabilities saved.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import Config
from data import load_gt, load_prepared, read_tsv
from evaluate import f05_entity
from matching import to_lists, write_lists
from model import assign_best, decide_expected_f, decide_threshold, predict_two, two_fold
from features import REC_COLS
from stage3 import STAGE3_FEATURES, SUP_COLS, _stage3_matrix, decide_threshold_src, support_features


def add_record_features(X3, meta, s1, pool):      # v4-era path (record features not yet in features.py)
    from features import record_features
    a, b = record_features(s1, pool)
    return np.hstack([X3, b[meta["p"].values], a[meta["q"].values]])


def _true_keys(cfg, s1, pool):
    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index) & gt["source1_entity_id"].isin(s1_pos.index)]
    gq, gp = s1_pos[gt["source1_entity_id"]].values, p_pos[gt["entity_id"]].values
    return gq, gp, gq.astype(np.int64) * len(pool) + gp


def _key(q, p, n):
    return np.asarray(q).astype(np.int64) * n + np.asarray(p)


def _rank_in_q(meta):
    return meta.groupby("q")["prob2"].rank(ascending=False, method="first").to_numpy()


def _eval_pr(sel, true_qp, qs):
    pred = {}
    for q, p in zip(sel["q"].values, sel["p"].values):
        pred.setdefault(q, set()).add(p)
    f = [f05_entity(pred.get(q, set()), true_qp.get(q, set())) for q in qs]
    tp = sum(len(pred.get(q, set()) & true_qp.get(q, set())) for q in qs)
    npred = sum(len(pred.get(q, set())) for q in qs)
    ntrue = sum(len(true_qp.get(q, set())) for q in qs)
    return float(np.mean(f)), tp / max(1, npred), tp / max(1, ntrue)


# ----------------------------------------------------------------------------- diag
def cmd_diag(cfg, a):
    s1, pool = load_prepared(cfg, "train")
    meta = pd.read_parquet(cfg.path("s3_train_meta.parquet"))
    gq, gp, tk = _true_keys(cfg, s1, pool)
    y = np.isin(_key(meta["q"].values, meta["p"].values, len(pool)), tk)
    rank = _rank_in_q(meta)
    nq = meta["q"].nunique()
    print(f"[diag] train meta {len(meta):,} pairs, {nq:,} entities, true pairs in candidates {int(y.sum()):,}", flush=True)
    for mp, mr in [(0.01, 8), (0.005, 10), (0.002, 10), (0.001, 10), (0.001, 12), (0.0, 12), (0.0, 20)]:
        sel = (meta["prob2"].values >= mp) & (rank <= mr)
        print(f"[diag] CE rule prob2>={mp} rank<={mr}: covers {y[sel].sum() / y.sum():.5f} of true pairs, "
              f"{sel.sum() / nq:.2f} pairs/entity", flush=True)

    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    idx = np.flatnonzero(meta["is_val"].values)
    X = np.load(cfg.path("s3_train_X.npy"), mmap_mode="r")
    m = meta.iloc[idx].reset_index(drop=True)
    X3 = _stage3_matrix(np.asarray(X[idx]), m, cfg.path("train_ce.parquet"))
    ms3 = [lgb.Booster(model_file=cfg.path(f"model_s3_{i}.txt")) for i in (0, 1)]
    m["prob"] = predict_two(ms3, X3, cfg.n_jobs)
    del X3
    m["y"] = y[idx]
    rep = json.load(open(cfg.path("stage3_report.json")))["decision"]
    best = assign_best(m["q"].values, m["p"].values, m["prob"].values).merge(m[["q", "p", "src", "y"]], on=["q", "p"])
    sel = decide_threshold_src(best, rep["p1"], rep["p2"]) if rep["rule"] == "src" else decide_threshold(best, rep["p1"])
    true_qp = {}
    for x, z in zip(gq, gp):
        if x in vset:
            true_qp.setdefault(x, set()).add(z)
    ctry = s1["country"].values
    for c in np.unique(ctry[val_q]):
        vq = val_q[ctry[val_q] == c]
        f, p_, r_ = _eval_pr(sel, true_qp, vq)
        print(f"[diag] val {c}: {len(vq):,} entities, F0.5 {f:.5f}, micro-P {p_:.4f}, micro-R {r_:.4f}", flush=True)
    n_true = sum(len(v) for v in true_qp.values())
    sel_keys = set(zip(sel["q"].values.tolist(), sel["p"].values.tolist()))
    mt = m[m["y"]]
    got = np.fromiter(((q, p) in sel_keys for q, p in zip(mt["q"].values, mt["p"].values)), bool, len(mt))
    lost = mt[~got]
    owner = best.set_index("p")["q"]
    lost_assign = owner.reindex(lost["p"].values).values != lost["q"].values
    ce = pd.read_parquet(cfg.path("train_ce.parquet"))
    not_ce = ~np.isin(_key(lost["q"].values, lost["p"].values, len(pool)), _key(ce["q"].values, ce["p"].values, len(pool)))
    print(f"[diag] val true pairs {n_true:,}: blocking-miss {(n_true - len(mt)) / n_true:.4f}, found {got.sum() / n_true:.4f}, "
          f"lost-to-other-entity {lost_assign.sum() / n_true:.4f}, lost-below-threshold {(~lost_assign).sum() / n_true:.4f}; "
          f"of lost pairs not CE-scored {not_ce.mean():.3f}", flush=True)
    print("[diag] lost pairs prob3 quantiles (10/25/50/75/90%)", np.quantile(lost["prob"].values, [.1, .25, .5, .75, .9]).round(3), flush=True)
    fp = sel.merge(m[["q", "p", "y"]], on=["q", "p"])
    fp = fp[~fp["y"]]
    print(f"[diag] val predicted {len(sel):,}, false {len(fp):,} ({len(fp) / len(sel):.4f}); FP prob quantiles",
          np.quantile(fp["prob"].values, [.1, .5, .9]).round(3), flush=True)

    ts1 = pd.read_parquet(cfg.path("test_s1.parquet"), columns=["country"])
    tm = pd.read_parquet(cfg.path("s3_test_meta.parquet"), columns=["q", "p", "prob2"])
    tm = tm.merge(pd.read_parquet(cfg.path("test_ce.parquet")), on=["q", "p"], how="left")
    tm["country"] = ts1["country"].values[tm["q"].values]
    for c, g in tm.groupby("country"):
        s = g[g["ce"].notna()]
        n = int((ts1["country"].values == c).sum())
        print(f"[diag] test {c}: {len(g) / n:.1f} pairs/ent, CE-scored {len(s) / n:.2f}/ent, "
              f"p2>=.5&ce<.2 {((s['prob2'] >= .5) & (s['ce'] < .2)).sum() / n:.4f}/ent, "
              f"p2<.2&ce>=.8 {((s['prob2'] < .2) & (s['ce'] >= .8)).sum() / n:.4f}/ent, "
              f"uncertain .2<=p2<.8 {((g['prob2'] >= .2) & (g['prob2'] < .8)).sum() / n:.4f}/ent, "
              f"mean ce|p2>=.9 {s['ce'][s['prob2'] >= .9].mean():.4f}, mean p2|ce>=.9 {s['prob2'][s['ce'] >= .9].mean():.4f}", flush=True)


# ----------------------------------------------------------------------------- pairs
def _pairs_test(cfg, a):
    ts1, _ = load_prepared(cfg, "test")
    tm = pd.read_parquet(cfg.path("s3_test_meta.parquet"))
    trank = _rank_in_q(tm)
    tneed = (tm["prob2"].values >= a.ce_min_prob) & (trank <= a.ce_max_rank)
    tm["need_ce"] = tneed
    tm.to_parquet(cfg.path("s4_test_meta.parquet"), index=False)
    tm.loc[tneed, ["q", "p"]].to_parquet(cfg.path("s4_test_ce_pairs.parquet"), index=False)
    print(f"[pairs] test: {int(tneed.sum()):,} CE pairs ({tneed.sum() / len(ts1):.2f}/entity)", flush=True)
    return tm


def cmd_pairs(cfg, a):
    t = time.time()
    if a.test_only:                     # synthetic world: test split only
        _pairs_test(cfg, a)
        return
    s1, pool = load_prepared(cfg, "train")
    if a.only_pseudo:
        return _pseudo(cfg, a, s1, pd.read_parquet(cfg.path("s4_test_meta.parquet")))
    meta = pd.read_parquet(cfg.path("s3_train_meta.parquet"))
    gq, gp, tk = _true_keys(cfg, s1, pool)
    y = np.isin(_key(meta["q"].values, meta["p"].values, len(pool)), tk)
    if a.all_train:
        miss = meta[SUP_COLS[0]].isna().to_numpy()
        if miss.any():
            sup = support_features(meta[miss].reset_index(drop=True), pool, cfg.n_jobs)
            for i, col in enumerate(SUP_COLS):
                meta.loc[miss, col] = sup[:, i]
        meta["in_s3"] = True
    rank = _rank_in_q(meta)
    need = (meta["prob2"].values >= a.ce_min_prob) & (rank <= a.ce_max_rank) & meta["in_s3"].values
    meta["need_ce"] = need
    meta.to_parquet(cfg.path("s4_train_meta.parquet"), index=False)
    meta.loc[need, ["q", "p"]].to_parquet(cfg.path("s4_train_ce_pairs.parquet"), index=False)
    print(f"[pairs] train: {int(need.sum()):,} CE pairs cover {y[need].sum() / y.sum():.5f} of true pairs", flush=True)

    # cross-encoder training list: every true pair + the stage-2 hardest negatives
    tr = meta[~meta["is_val"].values]
    ents = np.unique(tr["q"].values)
    rng = np.random.default_rng(11)
    ents = rng.choice(ents, size=min(a.ce_train_s1, len(ents)), replace=False)
    tr = tr[np.isin(tr["q"].values, ents)]
    ty = y[tr.index.values]
    pos = tr[ty]
    neg = tr[~ty].sort_values(["q", "prob2"], ascending=[True, False])
    r = neg.groupby("q").cumcount()
    hard = neg[r < a.neg_hard]
    rest = neg[r >= a.neg_hard].sample(frac=1.0, random_state=1).groupby("q").head(a.neg_rand)
    ce_train = pd.concat([pos[["q", "p"]].assign(y=1.0), hard[["q", "p"]].assign(y=0.0), rest[["q", "p"]].assign(y=0.0)],
                         ignore_index=True).sample(frac=1.0, random_state=2)
    ce_train.to_parquet(cfg.path("s4_ce_train_pairs.parquet"), index=False)
    print(f"[pairs] CE training pairs {len(ce_train):,} ({ce_train['y'].mean():.3f} positive) from {len(ents):,} entities", flush=True)

    tm = _pairs_test(cfg, a)
    _pseudo(cfg, a, s1, tm)
    print(f"[pairs] done in {time.time() - t:.0f}s", flush=True)


def _pseudo(cfg, a, s1, tm):
    """Pseudo-labels for countries that never appear in training, from v3's test predictions."""
    ts1, tpool = load_prepared(cfg, "test")
    unseen = sorted(set(ts1["country"].unique()) - set(s1["country"].unique()))
    tce = pd.read_parquet(cfg.path(a.ce_file))
    tm = tm.merge(tce, on=["q", "p"], how="left")
    tm["row"] = np.arange(len(tm))
    tm["rank"] = _rank_in_q(tm)
    tm["country"] = ts1["country"].values[tm["q"].values]
    if a.probs:
        # previous round's final probabilities (q, p, prob): a positive is the record's best entity
        # with prob >= pseudo_pos and the cross-encoder agreeing; negatives have prob <= pseudo_neg_p2
        pr = pd.read_parquet(a.probs)
        tm = tm.merge(pr, on=["q", "p"], how="left")
        tm["prob"] = tm["prob"].fillna(0)
        best = tm.sort_values("prob", ascending=False).drop_duplicates("p")
        best = best[best["prob"] >= a.pseudo_pos]
        mkeys = _key(best["q"].values, best["p"].values, len(tpool))
        tm["p2"] = tm["prob"]
    else:
        res = read_tsv(os.path.join(a.v3_out, "matching_results.tsv"))
        res["entity_id"] = res["matched_entity_ids"].str.split(",")
        res = res.explode("entity_id")
        res = res[res["entity_id"].notna() & (res["entity_id"] != "")]
        s1_pos = pd.Series(np.arange(len(ts1)), index=ts1["entity_id"].values)
        p_pos = pd.Series(np.arange(len(tpool)), index=tpool["entity_id"].values)
        mkeys = _key(s1_pos[res["source1_entity_id"]].values, p_pos[res["entity_id"]].values, len(tpool))
        tm["p2"] = tm["prob2"]
    for c in unseen:
        fr = tm[tm["country"] == c]
        matched = np.isin(_key(fr["q"].values, fr["p"].values, len(tpool)), mkeys)
        pos = fr[matched & (fr["p2"] >= a.pseudo_pos) & (fr["ce"] >= a.pseudo_ce)]
        # negatives: among the entity's top candidates, pairs stage 2 is very sure about (and the
        # cross-encoder, when it scored them, agrees); balanced against the positives so the
        # pseudo set does not teach "this country's text => match"
        neg = fr[~matched & (fr["rank"] <= a.ce_max_rank) & (fr["p2"] <= a.pseudo_neg_p2) &
                 (fr["ce"].isna() | (fr["ce"] <= a.pseudo_neg_ce))]
        neg = neg.sample(frac=1.0, random_state=3).groupby("q").head(a.pseudo_neg_per_q)
        if len(neg) > len(pos):
            neg = neg.sample(n=len(pos), random_state=4)
        ps = pd.concat([pos.assign(y=1.0), neg.assign(y=0.0)], ignore_index=True)[["row", "q", "p", "y"]]
        ps.to_parquet(cfg.path(f"s4_pseudo_{c}.parquet"), index=False)
        n_ent = int((ts1["country"].values == c).sum())
        print(f"[pairs] pseudo {c}: {len(pos):,} positives ({len(pos) / n_ent:.2f}/entity; v3 matched {matched.sum() / n_ent:.2f}/entity), "
              f"{len(neg):,} negatives", flush=True)


# ----------------------------------------------------------------------------- synthetic world
def _decide(best, key):
    return (decide_threshold_src(best, key[1], key[2]) if key[0] == "src" else
            decide_expected_f(best, key[1]) if key[0] == "expf" else decide_threshold(best, key[1]))


def _syn_rows(a):
    """Stage-3 rows of a synthetic labelled world (syn_world.py, prepared as a test split in a.syn):
    matrix, labels, meta, true pairs and the entities used (a.syn_max_s1 sample)."""
    cs = Config(data_dir=a.syn_data, work_dir=a.syn)
    s1s, pools = load_prepared(cs, "test")
    ms = pd.read_parquet(cs.path("s4_test_meta.parquet"))
    X3s = _stage3_matrix(np.asarray(np.load(cs.path("s3_test_X.npy"), mmap_mode="r")), ms,
                         cs.path(f"s4_test_ce{a.ce_suffix}.parquet"))
    g = read_tsv(os.path.join(a.syn_data, "syn_gt.tsv"))
    g["entity_id"] = g["matched_entity_ids"].str.split(",")
    g = g.explode("entity_id")
    g = g[g["entity_id"].notna() & (g["entity_id"] != "")]
    gq = pd.Series(np.arange(len(s1s)), index=s1s["entity_id"].values)[g["source1_entity_id"]].values
    gp = pd.Series(np.arange(len(pools)), index=pools["entity_id"].values)[g["entity_id"]].values
    true_qp = {}
    for x, z in zip(gq, gp):
        true_qp.setdefault(x, set()).add(z)
    qs = np.arange(len(s1s))
    if a.syn_max_s1 and len(s1s) > a.syn_max_s1:
        qs = np.sort(np.random.default_rng(5).choice(len(s1s), a.syn_max_s1, replace=False))
    rows = np.flatnonzero(np.isin(ms["q"].values, qs))
    ms = ms.iloc[rows].reset_index(drop=True)
    ys = np.isin(_key(ms["q"].values, ms["p"].values, len(pools)), _key(gq, gp, len(pools))).astype(np.int8)
    print(f"[syn] {len(qs):,} synthetic entities, {len(rows):,} rows ({ys.mean():.3f} positive), "
          f"true pairs in candidates {ys.sum() / max(1, sum(len(true_qp.get(q, ())) for q in qs)):.4f}", flush=True)
    return X3s[rows], ys, ms, true_qp, qs


def _syn_eval(ms, prob, true_qp, qs, key, tag):
    best = assign_best(ms["q"].values, ms["p"].values, prob).merge(ms[["q", "p", "src"]], on=["q", "p"])
    f, p_, r_ = _eval_pr(_decide(best, key), true_qp, qs)
    grid = {float(t): _eval_pr(decide_threshold(best, t), true_qp, qs)[0] for t in np.round(np.arange(0.3, 0.951, 0.05), 3)}
    tb = max(grid, key=grid.get)
    print(f"[syn-eval] {tag}: rule {key} F0.5 {f:.5f} (P {p_:.4f} R {r_:.4f}); best single threshold {tb} -> {grid[tb]:.5f}", flush=True)
    return {"f05": f, "precision": p_, "recall": r_, "best_thr": tb, "best_thr_f05": grid[tb]}


def cmd_syneval(cfg, a):
    """Score a synthetic world with this work dir's final models (not trained on it) and its decision rule."""
    import glob
    X3s, ys, ms, true_qp, qs = _syn_rows(a)
    models = [lgb.Booster(model_file=f) for f in sorted(glob.glob(cfg.path("model_s4_*.txt")))]
    prob = predict_two(models, X3s, cfg.n_jobs)
    d = json.load(open(cfg.path("stage4_report.json")))["decision"]
    key = (d["rule"], d["p1"], d["p2"])
    _syn_eval(ms, prob, true_qp, qs, key, f"{a.work_dir} models ({len(models)})")
    _syn_eval(ms, ms["prob2"].to_numpy(), true_qp, qs, key, "stage-2 prob")


# ----------------------------------------------------------------------------- fit
def _search_rules(meta, prob, val_q, true_qp):
    vset = set(val_q.tolist())
    best = assign_best(meta["q"].values, meta["p"].values, prob)
    best = best[best["q"].isin(vset)].merge(meta[["q", "p", "src"]], on=["q", "p"])
    res = {}
    grid = np.round(np.arange(0.4, 0.951, 0.025), 3)
    for thr in grid:
        res[("thr", float(thr), float(thr))] = _eval_pr(decide_threshold(best, thr), true_qp, val_q)
    for alpha in (0.0, 0.1):
        res[("expf", alpha, alpha)] = _eval_pr(decide_expected_f(best, alpha), true_qp, val_q)
    t0 = max((k for k in res if k[0] == "thr"), key=lambda k: res[k][0])[1]
    t2, t3 = t0, t0
    for _ in range(2):
        t2 = max(grid, key=lambda x: _eval_pr(decide_threshold_src(best, x, t3), true_qp, val_q)[0])
        t3 = max(grid, key=lambda x: _eval_pr(decide_threshold_src(best, t2, x), true_qp, val_q)[0])
    res[("src", float(t2), float(t3))] = _eval_pr(decide_threshold_src(best, t2, t3), true_qp, val_q)
    return res


def cmd_fit(cfg, a):
    t = time.time()
    s1, pool = load_prepared(cfg, "train")
    meta = pd.read_parquet(cfg.path("s4_train_meta.parquet"))
    gq, gp, tk = _true_keys(cfg, s1, pool)
    sub = meta["in_s3"].to_numpy()
    m3 = meta[sub].reset_index(drop=True)
    X = np.load(cfg.path("s3_train_X.npy"), mmap_mode="r")
    X3 = _stage3_matrix(np.asarray(X[sub]), m3, cfg.path(f"s4_train_ce{a.ce_suffix}.parquet"))
    del X
    if a.record_feats:
        X3 = add_record_features(X3, m3, s1, pool)
    feats = (STAGE3_FEATURES + REC_COLS) if a.record_feats else STAGE3_FEATURES
    y = np.isin(_key(m3["q"].values, m3["p"].values, len(pool)), tk).astype(np.int8)
    fold, is_val = m3["fold"].to_numpy(), m3["is_val"].to_numpy()
    print(f"[fit] train rows {len(m3):,} ({y.mean():.3f} positive), val rows {int(is_val.sum()):,}; {time.time() - t:.0f}s", flush=True)

    ts1, tpool = load_prepared(cfg, "test")
    tm = pd.read_parquet(cfg.path("s4_test_meta.parquet"))
    tX = np.load(cfg.path("s3_test_X.npy"), mmap_mode="r")
    tX3 = _stage3_matrix(np.asarray(tX), tm, cfg.path(f"s4_test_ce{a.ce_suffix}.parquet"))
    del tX
    if a.record_feats:
        tX3 = add_record_features(tX3, tm, ts1, tpool)
    print(f"[fit] test matrix {tX3.shape}; {time.time() - t:.0f}s", flush=True)
    n_pseudo = 0
    if a.pseudo:
        parts = [pd.read_parquet(p) for p in a.pseudo.split(",")]
        ps = pd.concat(parts, ignore_index=True)
        Xp = tX3[ps["row"].values]
        yp = ps["y"].values.astype(np.int8)
        fp = ((ps["row"].values.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)).astype(np.int64) % 2
        X3 = np.vstack([X3, Xp])
        y = np.concatenate([y, yp])
        fold = np.concatenate([fold, fp.astype(np.int8)])
        is_val = np.concatenate([is_val, np.zeros(len(yp), bool)])
        n_pseudo = len(ps)
        print(f"[fit] + {n_pseudo:,} pseudo-labelled rows ({yp.mean():.3f} positive)", flush=True)
    n_syn = 0
    if a.syn:
        X3s, ys, ms_syn, syn_true, syn_q = _syn_rows(a)
        fs = ((ms_syn["q"].values.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)).astype(np.int64) % 2
        X3 = np.vstack([X3, X3s])
        y = np.concatenate([y, ys])
        fold = np.concatenate([fold, fs.astype(np.int8)])
        is_val = np.concatenate([is_val, np.zeros(len(ys), bool)])
        n_syn = len(ys)
        del X3s
        print(f"[fit] + {n_syn:,} synthetic-world rows", flush=True)

    models, probs = [], []
    for s in range(a.seeds):
        params = dict(cfg.lgb_params, num_threads=cfg.n_jobs, seed=cfg.seed + s)
        ms, pr = two_fold(X3, y, fold, is_val, params, cfg.lgb_rounds, feats, cfg.n_jobs)
        models += ms
        probs.append(pr)
    for i, m in enumerate(models):
        m.save_model(cfg.path(f"model_s4_{i}.txt"))
    prob3 = np.mean(probs, axis=0)[:len(m3)]
    del X3

    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    true_qp = {}
    for x, z in zip(gq, gp):
        if x in vset:
            true_qp.setdefault(x, set()).add(z)
    prob_all = meta["prob2"].to_numpy().copy()
    prob_all[np.flatnonzero(sub)] = prob3
    pd.DataFrame({"q": meta["q"].values, "p": meta["p"].values, "src": meta["src"].values, "prob": prob_all}).to_parquet(
        cfg.path("val_scored_all.parquet"), index=False)          # every train pair, for post-hoc decision rules
    res = _search_rules(meta, prob_all, val_q, true_qp)
    cand = [k for k in res if a.rule == "auto" or k[0] == a.rule]
    key = max(cand, key=lambda k: res[k][0])
    for k in sorted(res, key=lambda k: -res[k][0])[:6]:
        print(f"[fit] {k}: F0.5 {res[k][0]:.5f} P {res[k][1]:.4f} R {res[k][2]:.4f}", flush=True)
    by_c = {}
    best = assign_best(meta["q"].values, meta["p"].values, prob_all)
    best = best[best["q"].isin(vset)].merge(meta[["q", "p", "src"]], on=["q", "p"])
    sel = (decide_threshold_src(best, key[1], key[2]) if key[0] == "src" else
           decide_expected_f(best, key[1]) if key[0] == "expf" else decide_threshold(best, key[1]))
    for c in np.unique(s1["country"].values[val_q]):
        vq = val_q[s1["country"].values[val_q] == c]
        by_c[c] = _eval_pr(sel, true_qp, vq)
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    rep = {"decision": {"rule": key[0], "p1": key[1], "p2": key[2]}, "val_macro_f05": res[key][0],
           "importance_top30": [(f, float(v)) for f, v in imp[:30]],
           "val_precision": res[key][1], "val_recall": res[key][2], "by_country": by_c, "seeds": a.seeds,
           "pseudo_rows": n_pseudo, "all": {"|".join(map(str, k)): v for k, v in res.items()}}
    if n_syn:
        ps = np.mean(probs, axis=0)[len(m3) + n_pseudo:]
        rep["syn_rows"] = n_syn
        rep["syn_oof"] = _syn_eval(ms_syn, ps, syn_true, syn_q, key, "out-of-fold (trained with the world)")
        rep["syn_stage2"] = _syn_eval(ms_syn, ms_syn["prob2"].to_numpy(), syn_true, syn_q, key, "stage-2 prob")
    json.dump(rep, open(cfg.path("stage4_report.json"), "w"), indent=1)
    print(f"[fit] chosen {key}: val F0.5 {res[key][0]:.5f} (P {res[key][1]:.4f} R {res[key][2]:.4f}) by country "
          f"{ {c: round(v[0], 5) for c, v in by_c.items()} }; {time.time() - t:.0f}s", flush=True)

    tprob = np.empty(len(tm), dtype=np.float32)
    step = 10_000_000
    for s in range(0, len(tm), step):
        tprob[s:s + step] = predict_two(models, tX3[s:s + step], cfg.n_jobs)
    del tX3
    pd.DataFrame({"q": tm["q"].values, "p": tm["p"].values, "prob": tprob}).to_parquet(cfg.path("test_scored_v4.parquet"), index=False)
    best = assign_best(tm["q"].values, tm["p"].values, tprob).merge(tm[["q", "p", "src"]], on=["q", "p"])
    sel = (decide_threshold_src(best, key[1], key[2]) if key[0] == "src" else
           decide_expected_f(best, key[1]) if key[0] == "expf" else decide_threshold(best, key[1]))
    s1_ids, pool_ids = ts1["entity_id"].values, tpool["entity_id"].values
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), s1_ids, to_lists(tm, s1_ids, pool_ids), "candidate_entity_ids")
    matches = to_lists(sel, s1_ids, pool_ids)
    write_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), s1_ids, matches, "matched_entity_ids")
    share = pd.Series(ts1["country"].values[np.unique(sel["q"].values)]).value_counts() / pd.Series(ts1["country"].values).value_counts()
    print(f"[fit] test: {len(matches):,} entities with matches, {len(sel):,} matched ids; share with matches {share.round(3).to_dict()}; "
          f"{time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "pairs", "fit", "syneval"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output_v4")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--v3-out", default="output_v3")
    ap.add_argument("--ce-min-prob", type=float, default=0.001)
    ap.add_argument("--ce-max-rank", type=int, default=10)
    ap.add_argument("--all-train", action="store_true")
    ap.add_argument("--ce-train-s1", type=int, default=400_000)
    ap.add_argument("--neg-hard", type=int, default=6)
    ap.add_argument("--neg-rand", type=int, default=2)
    ap.add_argument("--pseudo-pos", type=float, default=0.9)
    ap.add_argument("--pseudo-ce", type=float, default=0.9)
    ap.add_argument("--probs", default="", help="previous round's test probabilities (q,p,prob) for pseudo-labels")
    ap.add_argument("--ce-file", default="test_ce.parquet", help="cross-encoder score file used for pseudo-labels")
    ap.add_argument("--pseudo-neg-p2", type=float, default=0.005)
    ap.add_argument("--only-pseudo", action="store_true")
    ap.add_argument("--pseudo-neg-ce", type=float, default=0.05)
    ap.add_argument("--pseudo-neg-per-q", type=int, default=3)
    ap.add_argument("--pseudo", default="", help="comma list of s4_pseudo_*.parquet files to add to stage-3 training")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rule", default="auto", choices=["auto", "thr", "src", "expf"], help="force the decision-rule family")
    ap.add_argument("--ce-suffix", default="", help="use s4_{train,test}_ce<suffix>.parquet cross-encoder scores")
    ap.add_argument("--record-feats", action="store_true", help="v5: add record-level vocabulary / uniqueness features")
    ap.add_argument("--test-only", action="store_true", help="pairs: test split only (synthetic world)")
    ap.add_argument("--syn", default="", help="work dir of a prepared synthetic labelled world (syn_world.py) to add to training")
    ap.add_argument("--syn-data", default="dataset_syn", help="data dir of that world (test split + syn_gt.tsv)")
    ap.add_argument("--syn-max-s1", type=int, default=0, help="use at most this many synthetic entities")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, out_dir=a.out_dir, n_jobs=a.n_jobs)
    {"diag": cmd_diag, "pairs": cmd_pairs, "fit": cmd_fit, "syneval": cmd_syneval}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
