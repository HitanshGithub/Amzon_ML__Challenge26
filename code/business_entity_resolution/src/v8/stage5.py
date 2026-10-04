"""V8 stage 5 - a corrector over v7's probabilities on the uncertain band.

Measured ceilings on the held-out 10% of training entities:
    perfect scorer, current blocking            0.99731
    perfect per-entity cut on v7's ranking      0.99361
    v7 as submitted                             0.98928
so every point left is in the pair scorer, not in blocking (117 of 220,140 val entities have all
their true records blocked out) and not in the threshold. This stage therefore re-scores only the
band where v7 is unsure, using the new group-trained cross-encoder (ce8) next to v7's own score,
and leaves the 40% of pairs v7 already calls at >0.9995 / <0.0005 untouched.

  prep  assemble the feature frame for a split
  fit   2-fold LightGBM over entities, tune the decision rule on the val entities, write outputs
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from matching import to_lists, write_lists
from model import assign_best, decide_expected_f

BASE = ["prob", "prob2", "ce", "ce8", "d_ce8_ce", "d_ce8_prob", "d_ce_prob", "src", "ctry",
        "addr_empty", "n_q", "n_p", "n_q_src"]
SUP = ["sup_name_max", "sup_addr_max", "sup_prob_of_best", "sup_n_similar", "sup_weighted",
       "sup_other_src_name_max"]
CTX_SRC = ["prob", "ce", "ce8"]
CTX = [f"{w}_{s}_{k}" for s in CTX_SRC for w, k in
       [("q", "rank"), ("q", "gap"), ("q", "sum"), ("q", "second"), ("p", "rank"), ("p", "gap")]]
FEATS = BASE + SUP + CTX
CMAP = {"India": 0, "US": 1, "France": 2}


def _ctx(df):
    out = {}
    for s in CTX_SRC:
        v = df[s]
        gq, gp = df.groupby("q")[s], df.groupby("p")[s]
        out[f"q_{s}_rank"] = gq.rank(ascending=False, method="min").to_numpy(np.float32)
        mx = gq.transform("max")
        out[f"q_{s}_gap"] = (mx - v).to_numpy(np.float32)
        out[f"q_{s}_sum"] = gq.transform("sum").to_numpy(np.float32)
        srt = df[["q", s]].sort_values(["q", s], ascending=[True, False])
        sec = srt[srt.groupby("q").cumcount() == 1].set_index("q")[s]
        out[f"q_{s}_second"] = df["q"].map(sec).fillna(0).to_numpy(np.float32)
        out[f"p_{s}_rank"] = gp.rank(ascending=False, method="min").to_numpy(np.float32)
        out[f"p_{s}_gap"] = (gp.transform("max") - v).to_numpy(np.float32)
    return out


def build(cfg, split, a):
    t = time.time()
    s1, pool = load_prepared(cfg, split)
    band = pd.read_parquet(a.band if split == "train" else a.band_test)
    if split == "train":
        d = pd.read_parquet(cfg.path(a.probs))[["q", "p", "src", "prob", "prob2", "y"]]
    else:
        meta = pd.read_parquet(cfg.path("s4_test_meta.parquet"), columns=["q", "p", "src", "prob2"] + SUP)
        pr = pd.read_parquet(cfg.path("test_scored_v4.parquet"))
        d = meta.merge(pr, on=["q", "p"], how="left")
        del meta, pr
    print(f"[s5:{split}] all pairs {len(d):,}; band {len(band):,}; {time.time()-t:.0f}s", flush=True)
    full = d[["q", "p", "src", "prob"]].copy()          # kept to fill the untouched pairs later
    d = band.merge(d, on=["q", "p"], how="inner")
    ce = pd.read_parquet(cfg.path(f"s4_{split}_ce_ensfr.parquet"))
    d = d.merge(ce, on=["q", "p"], how="left"); del ce
    parts = [pd.read_parquet(p) for p in (a.ce8 if split == "train" else a.ce8_test).split(",")]
    c8 = parts[0]
    for k, extra in enumerate(parts[1:], 1):
        c8 = c8.merge(extra.rename(columns={"ce8": f"ce8_{k}"}), on=["q", "p"], how="outer")
    if len(parts) > 1:
        cols = ["ce8"] + [f"ce8_{k}" for k in range(1, len(parts))]
        c8["ce8"] = c8[cols].mean(axis=1)
        c8 = c8[["q", "p", "ce8"]]
    d = d.merge(c8, on=["q", "p"], how="left"); del c8, parts
    if split == "train":
        sup = pd.read_parquet(cfg.path("v8_train_sup.parquet"))
        d = d.merge(sup, on=["q", "p"], how="left"); del sup
    print(f"[s5:{split}] band rows {len(d):,}, ce8 present {d['ce8'].notna().mean():.4f}; "
          f"{time.time()-t:.0f}s", flush=True)
    d["ctry"] = pd.Series(s1["country"].values[d["q"].to_numpy()]).map(CMAP).fillna(-1).to_numpy(np.int8)
    d["addr_empty"] = pool["addr_empty"].values[d["p"].to_numpy()].astype(np.int8)
    d["d_ce8_ce"] = (d["ce8"] - d["ce"]).astype(np.float32)
    d["d_ce8_prob"] = (d["ce8"] - d["prob"]).astype(np.float32)
    d["d_ce_prob"] = (d["ce"] - d["prob"]).astype(np.float32)
    d["n_q"] = d.groupby("q")["p"].transform("size").to_numpy(np.int32)
    d["n_p"] = d.groupby("p")["q"].transform("size").to_numpy(np.int32)
    d["n_q_src"] = d.groupby(["q", "src"])["p"].transform("size").to_numpy(np.int32)
    for k, v in _ctx(d).items():
        d[k] = v
    print(f"[s5:{split}] features done; {time.time()-t:.0f}s", flush=True)
    return s1, pool, d, full


def macro(sel, true_qp, val_q):
    pred = {}
    for q, p in zip(sel["q"].to_numpy(), sel["p"].to_numpy()):
        pred.setdefault(q, set()).add(p)
    f = tp = npred = ntrue = 0.0
    for q in val_q:
        pr, tr = pred.get(q, set()), true_qp.get(q, set())
        f += f05_entity(pr, tr)
        tp += len(pr & tr); npred += len(pr); ntrue += len(tr)
    return f / len(val_q), tp / max(1.0, npred), tp / max(1.0, ntrue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work7")
    ap.add_argument("--out-dir", default="output_v8")
    ap.add_argument("--probs", default="val_scored_all_v7.parquet")
    ap.add_argument("--band", default="work/v8_train_band.parquet")
    ap.add_argument("--band-test", default="work/v8_test_band.parquet")
    ap.add_argument("--ce8", default="work/v8_train_ce8.parquet")
    ap.add_argument("--ce8-test", default="work/v8_test_ce8.parquet")
    ap.add_argument("--rounds", type=int, default=900)
    ap.add_argument("--n-jobs", type=int, default=31)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--drop-eval", type=float, default=0.0,
                    help="simulate the test set's decoy density: pretend this share of train S1 "
                         "entities does not exist, so their records become unowned distractors "
                         "competing for the val entities (test pool/S1 = 5.75 vs train 4.68)")
    a = ap.parse_args()
    t = time.time()
    cfg = Config(data_dir="dataset", work_dir=a.work_dir, out_dir=a.out_dir, n_jobs=a.n_jobs)
    s1, pool, d, full = build(cfg, "train", a)

    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    gt = load_gt(cfg)
    sp = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    pp = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(pp.index)]
    gq, gp = sp[gt["source1_entity_id"]].values, pp[gt["entity_id"]].values
    true_qp = {}
    for q, p in zip(gq, gp):
        if q in vset:
            true_qp.setdefault(q, set()).add(p)

    drop_q = np.zeros(len(s1), bool)
    if a.drop_eval > 0:
        share = a.drop_eval / (1.0 - cfg.val_s1_frac)
        drop_q = (u >= cfg.val_s1_frac) & (u < cfg.val_s1_frac + share * (1.0 - cfg.val_s1_frac))
        print(f"[s5] decoy simulation: {int(drop_q.sum()):,} entities treated as non-existent "
              f"({drop_q.mean():.3f} of all)", flush=True)
    is_val = np.isin(d["q"].to_numpy(), val_q)
    fold = ((d["q"].to_numpy().astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)).astype(np.int64) % 2
    X = d[FEATS].to_numpy(np.float32)
    y = d["y"].to_numpy(np.int8)
    print(f"[s5] X {X.shape}; band positives {y.mean():.4f}; val rows {int(is_val.sum()):,}; "
          f"{time.time()-t:.0f}s", flush=True)
    params = dict(cfg.lgb_params, num_threads=cfg.n_jobs, seed=cfg.seed, learning_rate=0.05,
                  num_leaves=127, min_data_in_leaf=200)
    models, oof = [], np.zeros(len(d), np.float32)
    for f in (0, 1):
        rows = (fold == f) & ~is_val
        m = lgb.train(params, lgb.Dataset(X[rows], y[rows], feature_name=FEATS), a.rounds,
                      valid_sets=[lgb.Dataset(X[is_val], y[is_val])],
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(300)])
        other = (fold != f) & ~is_val
        oof[other] = m.predict(X[other], num_threads=cfg.n_jobs)
        oof[is_val] += 0.5 * m.predict(X[is_val], num_threads=cfg.n_jobs)
        models.append(m)
        print(f"[s5] fold{f} best_iter {m.best_iteration} logloss "
              f"{m.best_score['valid_0']['binary_logloss']:.5f}; {time.time()-t:.0f}s", flush=True)
    for i, m in enumerate(models):
        m.save_model(cfg.path(f"model_s5_{i}.txt"))
    imp = sorted(zip(FEATS, models[0].feature_importance("gain")), key=lambda x: -x[1])[:22]
    print("[s5] importance " + ", ".join(f"{k}:{v/1e3:.0f}k" for k, v in imp), flush=True)

    # ---- merge corrected band probabilities back into the full pair table
    corr = pd.DataFrame({"q": d["q"].to_numpy(), "p": d["p"].to_numpy(), "pc": oof})
    fu = full.merge(corr, on=["q", "p"], how="left")
    pf = np.where(np.isnan(fu["pc"].to_numpy()), fu["prob"].to_numpy(), fu["pc"].to_numpy()).astype(np.float32)
    rep = {}
    keep_row = ~drop_q[fu["q"].to_numpy()] if a.drop_eval > 0 else np.ones(len(fu), bool)
    for label, pv in (("v7", fu["prob"].to_numpy()), ("v8", pf)):
        best = assign_best(fu["q"].to_numpy()[keep_row], fu["p"].to_numpy()[keep_row], pv[keep_row])
        best = best.merge(fu[["q", "p", "src"]], on=["q", "p"], how="left")
        bv = best[best["q"].isin(vset)].reset_index(drop=True)
        grid = np.round(np.arange(0.40, 0.951, 0.05), 3)
        res = {}
        for t2 in grid:
            for t3 in grid:
                if abs(t2 - t3) > 0.10:
                    continue
                thr = np.where(bv["src"].to_numpy() == 2, t2, t3)
                res[(float(t2), float(t3))] = macro(bv[bv["prob"].to_numpy() >= thr], true_qp, val_q)
        for alpha in (0.0, 0.05, 0.1, 0.2):     # plug-in expected-F0.5 cut per entity
            res[("expf", alpha)] = macro(decide_expected_f(bv, alpha), true_qp, val_q)
        k = max((x for x in res if x[0] != "expf"), key=lambda x: res[x][0])
        kk_all = max(res, key=lambda x: res[x][0])
        rep[label] = {"best_thr": list(k), "f05": res[k][0], "P": res[k][1], "R": res[k][2],
                      "best_any": [str(kk_all), res[kk_all][0]]}
        byc = {}
        for c in np.unique(s1["country"].values[val_q]):
            vq = val_q[s1["country"].values[val_q] == c]
            bf, bkc = -1.0, None
            for t2 in grid:
                for t3 in grid:
                    if abs(t2 - t3) > 0.10:
                        continue
                    thr = np.where(bv["src"].to_numpy() == 2, t2, t3)
                    f = macro(bv[bv["prob"].to_numpy() >= thr], true_qp, vq)[0]
                    if f > bf:
                        bf, bkc = f, (float(t2), float(t3))
            thr = np.where(bv["src"].to_numpy() == 2, k[0], k[1])
            byc[c] = list(macro(bv[bv["prob"].to_numpy() >= thr], true_qp, vq)) + [bf, list(bkc)]
        rep[label]["by_country"] = byc
        print(f"[s5] {label}: best thr {k} F0.5 {res[k][0]:.5f} P {res[k][1]:.4f} R {res[k][2]:.4f} "
              f"by-country { {c: round(v[0], 5) for c, v in byc.items()} }; {time.time()-t:.0f}s", flush=True)
        if label == "v8":
            rep["v8_grid_top"] = [[str(kk), res[kk][0]] for kk in sorted(res, key=lambda x: -res[x][0])[:10]]
    json.dump(rep, open(cfg.path(f"stage5_report_{a.tag}.json"), "w"), indent=1)
    thr_key = rep["v8"]["best_thr"]
    del X, d, fu, corr, full
    if a.no_test:
        print("STAGE5_DONE", flush=True)
        return

    # ---- test
    ts1, tpool, td, tfull = build(cfg, "test", a)
    tX = td[FEATS].to_numpy(np.float32)
    tp_ = np.mean([m.predict(tX, num_threads=cfg.n_jobs) for m in models], axis=0).astype(np.float32)
    corr = pd.DataFrame({"q": td["q"].to_numpy(), "p": td["p"].to_numpy(), "pc": tp_})
    del tX, td
    fu = tfull.merge(corr, on=["q", "p"], how="left")
    pf = np.where(np.isnan(fu["pc"].to_numpy()), fu["prob"].to_numpy(), fu["pc"].to_numpy()).astype(np.float32)
    pd.DataFrame({"q": fu["q"].to_numpy(), "p": fu["p"].to_numpy(), "prob": pf}).to_parquet(
        cfg.path(f"test_scored_{a.tag}.parquet"), index=False)
    best = assign_best(fu["q"].to_numpy(), fu["p"].to_numpy(), pf).merge(fu[["q", "p", "src"]], on=["q", "p"], how="left")
    thr = np.where(best["src"].to_numpy() == 2, thr_key[0], thr_key[1])
    sel = best[best["prob"].to_numpy() >= thr]
    os.makedirs(cfg.out_dir, exist_ok=True)
    s1_ids, pool_ids = ts1["entity_id"].values, tpool["entity_id"].values
    write_lists(os.path.join(cfg.out_dir, "candidate_pairs.tsv"), s1_ids,
                to_lists(tfull, s1_ids, pool_ids), "candidate_entity_ids")
    write_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), s1_ids,
                to_lists(sel, s1_ids, pool_ids), "matched_entity_ids")
    share = (pd.Series(ts1["country"].values[np.unique(sel["q"].to_numpy())]).value_counts()
             / pd.Series(ts1["country"].values).value_counts())
    print(f"[s5] test: {len(sel):,} matched ids; share with matches {share.round(3).to_dict()}; "
          f"{time.time()-t:.0f}s", flush=True)
    print("STAGE5_DONE", flush=True)


if __name__ == "__main__":
    main()
