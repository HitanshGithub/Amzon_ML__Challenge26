"""V8 decision lab - post-hoc rules on top of v7's probabilities, scored on the held-out 10%.

v7's recall loss splits into: 1.71% of true pairs scored below the threshold, 0.49% handed to the
wrong S1 entity, 0.90% never blocked. The first two are decision-stage losses, so they can be
attacked without retraining. Levers, each tuned here and kept only if macro F0.5 improves:

  source-completion  P(>=1 S3 match | >=1 S2 match) = 0.9255 in the ground truth (and symmetric)
                     against a 0.879 marginal. An entity that ended up with S2 matches but no S3
                     match is probably missing one -> lower the bar for its best S3 candidate.
  sibling rescue     the S2/S3 copies of one business resemble each other more than they resemble
                     the S1 record (different sources, different noise). A rejected candidate that
                     is near-identical to one we already accepted for this entity is very likely
                     the same business -> accept it below threshold.
  per-country thr    France is absent here, but India/US already want different cut-offs.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
from stage3 import decide_threshold_src


def macro(sel_q, sel_p, true_qp, val_q):
    pred = {}
    for q, p in zip(sel_q, sel_p):
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
    ap.add_argument("--probs", default="val_scored_all_v7.parquet")
    ap.add_argument("--out", default="v8/lab_report.json")
    ap.add_argument("--cand-out", default="v8_rescue_cand.parquet")
    ap.add_argument("--n-jobs", type=int, default=31)
    a = ap.parse_args()
    t = time.time()
    cfg = Config(data_dir="dataset", work_dir=a.work_dir, n_jobs=a.n_jobs)
    s1, pool = load_prepared(cfg, "train")
    ctry = s1["country"].values
    d = pd.read_parquet(cfg.path(a.probs))
    print(f"[lab] {len(d):,} scored pairs; {time.time()-t:.0f}s", flush=True)

    u = np.random.default_rng(cfg.seed).random(len(s1))
    val_q = np.flatnonzero(u < cfg.val_s1_frac)
    vset = set(val_q.tolist())
    # ---- ground truth for the val entities (ALL true pairs, blocking misses included: identical
    # to how the v7 report measured recall, so numbers are comparable)
    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index)]
    gq, gp = s1_pos[gt["source1_entity_id"]].values, p_pos[gt["entity_id"]].values
    true_qp = {}
    for q, p in zip(gq, gp):
        if q in vset:
            true_qp.setdefault(q, set()).add(p)
    print(f"[lab] val entities {len(val_q):,}, true pairs {sum(map(len, true_qp.values())):,}; "
          f"{time.time()-t:.0f}s", flush=True)

    # ---- one owner per pool record (zero id reuse across entities in the ground truth)
    best = assign_best(d["q"].values, d["p"].values, d["prob"].values)
    best = best.merge(d[["q", "p", "src"]], on=["q", "p"], how="left")
    print(f"[lab] assigned {len(best):,} records; {time.time()-t:.0f}s", flush=True)
    bv = best[best["q"].isin(vset)].reset_index(drop=True)
    del best
    rep = {}

    def score(name, sel):
        r = macro(sel["q"].values, sel["p"].values, true_qp, val_q)
        rep[name] = [float(x) for x in r]
        print(f"[lab] {name:46s} F0.5 {r[0]:.5f}  P {r[1]:.4f}  R {r[2]:.4f}", flush=True)
        return r

    thr0 = np.where(bv["src"].to_numpy() == 2, 0.75, 0.7)
    base = bv[bv["prob"].to_numpy() >= thr0].reset_index(drop=True)
    score("baseline src(0.75,0.70)", base)

    # ---- per-country thresholds
    cv = ctry[bv["q"].values]
    for c in np.unique(cv):
        vq = val_q[ctry[val_q] == c]
        sub = bv[cv == c]
        bf, bk = -1.0, None
        for t2 in np.arange(0.55, 0.93, 0.05):
            for t3 in np.arange(0.55, 0.93, 0.05):
                s = decide_threshold_src(sub, t2, t3)
                f = macro(s["q"].values, s["p"].values, true_qp, vq)[0]
                if f > bf:
                    bf, bk = f, (round(float(t2), 3), round(float(t3), 3))
        rep[f"country_thr_{c}"] = [bf, list(bk)]
        print(f"[lab] per-country thr {c}: {bk} F0.5 {bf:.5f} ({time.time()-t:.0f}s)", flush=True)

    # ---- rescue candidate set: val rows this entity owns but the threshold rejected
    acc = base.groupby("q")["p"].apply(list).to_dict()
    cnt = base.pivot_table(index="q", columns="src", values="p", aggfunc="count").fillna(0)
    has2 = set(cnt.index[cnt.get(2, pd.Series(0.0, index=cnt.index)) > 0].tolist())
    has3 = set(cnt.index[cnt.get(3, pd.Series(0.0, index=cnt.index)) > 0].tolist())
    print(f"[lab] >=1 S2 {len(has2):,}  >=1 S3 {len(has3):,}  S2-only {len(has2 - has3):,}  "
          f"S3-only {len(has3 - has2):,}; {time.time()-t:.0f}s", flush=True)

    cand = bv[(bv["prob"].to_numpy() < thr0) & (bv["prob"].to_numpy() >= 0.02)].reset_index(drop=True)
    print(f"[lab] rescue candidates {len(cand):,}; {time.time()-t:.0f}s", flush=True)

    from rapidfuzz import fuzz
    nk, ak = pool["name_key"].values, pool["addr_norm"].values
    cq, cp = cand["q"].to_numpy(), cand["p"].to_numpy()
    sib = np.zeros(len(cand), np.float32); sib_a = np.zeros(len(cand), np.float32)
    for i in range(len(cand)):
        lst = acc.get(cq[i])
        if not lst:
            continue
        bn, ba = nk[cp[i]], ak[cp[i]]
        m = ma = 0.0
        for pj in lst:
            s = fuzz.token_set_ratio(bn, nk[pj])
            if s > m:
                m = s
            if ba and ak[pj]:
                s = fuzz.token_set_ratio(ba, ak[pj])
                if s > ma:
                    ma = s
        sib[i], sib_a[i] = m / 100.0, ma / 100.0
        if i and i % 500_000 == 0:
            print(f"[lab] sib {i:,}/{len(cand):,} {time.time()-t:.0f}s", flush=True)
    cand["sib"] = sib; cand["sib_a"] = sib_a
    src2 = cand["src"].to_numpy() == 2
    need = np.where(src2, np.fromiter(((q in has3 and q not in has2) for q in cq), bool, len(cq)),
                    np.fromiter(((q in has2 and q not in has3) for q in cq), bool, len(cq)))
    cand["need"] = need
    cand["rk"] = cand.groupby(["q", "src"])["prob"].rank(ascending=False, method="first").to_numpy()
    cand.to_parquet(cfg.path(a.cand_out), index=False)
    print(f"[lab] need-rows {int(need.sum()):,}; saved candidates; {time.time()-t:.0f}s", flush=True)

    for pmin in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        add = cand[need & (cand["rk"].to_numpy() == 1) & (cand["prob"].to_numpy() >= pmin)]
        score(f"+src-completion pmin={pmin}", pd.concat([base, add], ignore_index=True))
    for smin in (0.88, 0.92, 0.96, 1.00):
        for pmin in (0.05, 0.10, 0.20, 0.35):
            add = cand[(cand["sib"].to_numpy() >= smin) & (cand["prob"].to_numpy() >= pmin)]
            score(f"+sibling s>={smin} p>={pmin}", pd.concat([base, add], ignore_index=True))
    for smin in (0.92, 0.96):
        for amin in (0.9, 1.0):
            for pmin in (0.05, 0.15):
                add = cand[(cand["sib"].to_numpy() >= smin) & (cand["sib_a"].to_numpy() >= amin)
                           & (cand["prob"].to_numpy() >= pmin)]
                score(f"+sib+addr s>={smin} a>={amin} p>={pmin}", pd.concat([base, add], ignore_index=True))
    json.dump(rep, open(a.out, "w"), indent=1)
    print("LAB_DONE", flush=True)


if __name__ == "__main__":
    main()
