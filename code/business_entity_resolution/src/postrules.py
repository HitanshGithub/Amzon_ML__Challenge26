"""Post-hoc decision rules on top of a stage-4 model's probabilities.

  tune   empty-list rescue: a validation S1 entity the decision rule left without matches gets its best
         owned candidate when prob >= lo. lo is chosen on one half of the validation entities and
         checked on the other half; the rescue is used only if it gains on both halves. Also reports
         rule A on validation.
  apply  final file: the model's decision (+ the rescue if it passed), then rule A: drop matches whose
         pool record has an empty address while the entity's normalised name is shared by another S1
         entity of the same country - such records cannot be told apart, and F0.5 favours leaving
         them out.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import Config
from data import load_prepared
from features import record_features
from matching import to_lists, write_lists
from model import assign_best
from stage4 import _decide, _eval_pr, _true_keys

GRID = np.round(np.arange(0.30, 0.701, 0.05), 3)


def _key(rep):
    d = rep["decision"]
    return d["rule"], d["p1"], d["p2"]


def _rescue(best, sel, qs, lo):
    empty = np.setdiff1d(qs, sel["q"].values)
    cand = best[best["q"].isin(empty)].sort_values("prob", ascending=False).drop_duplicates("q")
    return pd.concat([sel, cand[cand["prob"] >= lo][["q", "p", "prob"]]], ignore_index=True)


def _rule_a(sel, s1, pool):
    ra, _ = record_features(s1, pool)
    q, p = sel["q"].values, sel["p"].values
    drop = (pool["addr_empty"].values[p] == 1) & (ra[q, 1] > 1)
    return sel[~drop], drop


def cmd_tune(cfg, a):
    s1, pool = load_prepared(cfg, "train")
    gq, gp, _ = _true_keys(cfg, s1, pool)
    key = _key(json.load(open(cfg.path("stage4_report.json"))))
    v = pd.read_parquet(cfg.path("val_scored_all.parquet"))
    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    true_qp = {}
    for x, z in zip(gq, gp):
        if x in vset:
            true_qp.setdefault(x, set()).add(z)
    best = assign_best(v["q"].values, v["p"].values, v["prob"].values).merge(v[["q", "p", "src"]], on=["q", "p"])
    best = best[best["q"].isin(vset)]
    sel = _decide(best, key)
    half = ((val_q.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)).astype(np.int64) % 2
    qa, qb = val_q[half == 0], val_q[half == 1]
    f = lambda s, qs: _eval_pr(s, true_qp, qs)[0]
    base = {"all": f(sel, val_q), "a": f(sel, qa), "b": f(sel, qb)}
    res = {}
    for lo in GRID:
        s2 = _rescue(best, sel, val_q, lo)
        res[float(lo)] = {"all": f(s2, val_q) - base["all"], "a": f(s2, qa) - base["a"], "b": f(s2, qb) - base["b"], "added": len(s2) - len(sel)}
        print(f"[tune] rescue lo={lo}: +{res[float(lo)]['added']:,} pairs, dF all {res[float(lo)]['all']:+.5f} "
              f"(half A {res[float(lo)]['a']:+.5f}, half B {res[float(lo)]['b']:+.5f})", flush=True)
    lo_a = max(res, key=lambda k: res[k]["a"])
    lo_b = max(res, key=lambda k: res[k]["b"])
    passed = res[lo_a]["b"] > 0 and res[lo_b]["a"] > 0
    lo = max(res, key=lambda k: res[k]["all"])
    s_final = _rescue(best, sel, val_q, lo) if passed else sel
    s_a, drop = _rule_a(s_final, s1, pool)
    out = {"decision": key, "base_f05": base["all"], "rescue": {"lo": lo if passed else None, "lo_a": lo_a, "lo_b": lo_b,
           "gain_b_at_lo_a": res[lo_a]["b"], "gain_a_at_lo_b": res[lo_b]["a"], "passed": passed, "gain_all": res[lo]["all"] if passed else 0.0},
           "rule_a": {"dropped": int(drop.sum()), "gain": f(s_a, val_q) - f(s_final, val_q)}, "final_f05": f(s_a, val_q), "grid": res}
    json.dump(out, open(cfg.path("postrules_report.json"), "w"), indent=1)
    print(f"[tune] base {base['all']:.5f}; rescue {'PASSED lo=' + str(lo) if passed else 'rejected'} "
          f"(lo_A {lo_a} -> B {res[lo_a]['b']:+.5f}, lo_B {lo_b} -> A {res[lo_b]['a']:+.5f}); "
          f"rule A drops {int(drop.sum()):,} ({out['rule_a']['gain']:+.5f}); final {out['final_f05']:.5f}", flush=True)


def cmd_apply(cfg, a):
    s1, pool = load_prepared(cfg, "test")
    key = _key(json.load(open(cfg.path("stage4_report.json"))))
    pr = json.load(open(cfg.path("postrules_report.json")))
    sc = pd.read_parquet(cfg.path("test_scored_v4.parquet"))
    sc["src"] = pool["src"].values[sc["p"].values]
    best = assign_best(sc["q"].values, sc["p"].values, sc["prob"].values).merge(sc[["q", "p", "src"]], on=["q", "p"])
    sel = _decide(best, key)
    n0 = len(sel)
    lo = pr["rescue"]["lo"]
    if lo is not None and not a.no_rescue:
        sel = _rescue(best, sel, np.arange(len(s1)), lo)
    n1 = len(sel)
    sel, drop = _rule_a(sel, s1, pool)
    s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
    os.makedirs(a.out_dir, exist_ok=True)
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, to_lists(sel, s1_ids, pool_ids), "matched_entity_ids")
    ctry = s1["country"].values
    print(f"[apply] model decision {n0:,} pairs; rescue lo={lo if not a.no_rescue else None} +{n1 - n0:,}; rule A -{int(drop.sum()):,}; "
          f"final {len(sel):,}; matched ids per S1 by country "
          f"{(pd.Series(ctry[sel['q'].values]).value_counts() / pd.Series(ctry).value_counts()).round(3).to_dict()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "apply"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work7")
    ap.add_argument("--out-dir", default="output_final")
    ap.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    ap.add_argument("--no-rescue", action="store_true")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir, out_dir=a.out_dir, n_jobs=a.n_jobs, drop_s1_frac=0.2)
    cmd_tune(cfg, a) if a.cmd == "tune" else cmd_apply(cfg, a)


if __name__ == "__main__":
    main()
