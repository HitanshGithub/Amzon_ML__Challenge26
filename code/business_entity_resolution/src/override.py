"""Blend an external pair scorer (the LoRA-tuned 7B classifier) into the final decisions.

  valprobs : stage-3 probabilities of the final model for every validation pair (saved once)
  eval     : for the validation pairs the external scorer saw, try prob' = (1-w)*prob + w*p_ext and
             report macro-F0.5 over the validation entities for each w (same assignment + rule)
  apply    : apply the chosen w to the test probabilities and write matching_results.tsv
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import Config
from data import load_prepared
from matching import to_lists, write_lists
from model import assign_best, predict_two
from stage3 import _stage3_matrix, decide_threshold_src
from stage4 import _eval_pr, _key, _true_keys


def cmd_valprobs(cfg, a):
    s1, pool = load_prepared(cfg, "train")
    meta = pd.read_parquet(cfg.path("s4_train_meta.parquet"))
    idx = np.flatnonzero(meta["is_val"].values)
    X = np.load(cfg.path("s3_train_X.npy"), mmap_mode="r")
    m = meta.iloc[idx].reset_index(drop=True)
    X3 = _stage3_matrix(np.asarray(X[idx]), m, cfg.path(f"s4_train_ce{a.ce_suffix}.parquet"))
    models = [lgb.Booster(model_file=cfg.path(f"model_s4_{i}.txt")) for i in range(a.n_models)]
    m["prob"] = predict_two(models, X3, cfg.n_jobs)
    m[["q", "p", "src", "prob"]].to_parquet(cfg.path("val_scored_final.parquet"), index=False)
    print(f"[override] validation pairs {len(m):,}", flush=True)


def _blend(df, ext, w):
    d = df.merge(ext, on=["q", "p"], how="left")
    has = d["p_ext"].notna().to_numpy()
    prob = d["prob"].to_numpy().copy()
    prob[has] = (1 - w) * prob[has] + w * d["p_ext"].to_numpy()[has]
    return d.assign(prob=prob), has


def cmd_eval(cfg, a):
    s1, pool = load_prepared(cfg, "train")
    val = pd.read_parquet(cfg.path("val_scored_final.parquet"))
    ext = pd.read_parquet(a.ext)[["q", "p", a.ext_col]].rename(columns={a.ext_col: "p_ext"})
    gq, gp, _ = _true_keys(cfg, s1, pool)
    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    true_qp = {}
    for x, z in zip(gq, gp):
        if x in vset:
            true_qp.setdefault(x, set()).add(z)
    touched = np.unique(val.merge(ext, on=["q", "p"])["q"].values)
    print(f"[override] external scores on {len(ext):,} pairs, touching {len(touched):,} validation entities", flush=True)
    res = {}
    for w in a.weights:
        d, _ = _blend(val, ext, w)
        best = assign_best(d["q"].values, d["p"].values, d["prob"].values).merge(d[["q", "p", "src"]], on=["q", "p"])
        sel = decide_threshold_src(best, a.thr, a.thr3)
        full = _eval_pr(sel, true_qp, val_q)
        part = _eval_pr(sel[sel["q"].isin(set(touched.tolist()))], true_qp, touched)
        res[w] = (full, part)
        print(f"[override] w={w:.2f}: all-val F0.5 {full[0]:.5f} (P {full[1]:.4f} R {full[2]:.4f}) | touched entities F0.5 {part[0]:.5f}", flush=True)
    wbest = max(res, key=lambda w: res[w][0][0])
    json.dump({"best_w": wbest, "results": {str(k): v for k, v in res.items()}}, open(cfg.path("override_report.json"), "w"), indent=1)
    print(f"[override] best w = {wbest}", flush=True)


def cmd_apply(cfg, a):
    s1, pool = load_prepared(cfg, "test")
    sc = pd.read_parquet(a.scored)
    if "src" not in sc.columns:
        sc["src"] = pool["src"].values[sc["p"].values]
    ext = pd.read_parquet(a.ext)[["q", "p", a.ext_col]].rename(columns={a.ext_col: "p_ext"})
    d, has = _blend(sc, ext, a.w)
    best = assign_best(d["q"].values, d["p"].values, d["prob"].values).merge(d[["q", "p", "src"]], on=["q", "p"])
    sel = decide_threshold_src(best, a.thr, a.thr3)
    s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
    os.makedirs(a.out_dir, exist_ok=True)
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, to_lists(sel, s1_ids, pool_ids), "matched_entity_ids")
    cnt = pd.Series(s1["country"].values[sel["q"].values]).value_counts().to_dict()
    print(f"[override] blended {int(has.sum()):,} test pairs with w={a.w}; matched ids per country {cnt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["valprobs", "eval", "apply"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work6")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--ce-suffix", default="_fullens")
    ap.add_argument("--n-models", type=int, default=6)
    ap.add_argument("--ext", default="")
    ap.add_argument("--ext-col", default="p_lora")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--thr3", type=float, default=0.725)
    ap.add_argument("--scored", default="")
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--out-dir", default="output_final")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, n_jobs=a.n_jobs, drop_s1_frac=0.2)
    {"valprobs": cmd_valprobs, "eval": cmd_eval, "apply": cmd_apply}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
