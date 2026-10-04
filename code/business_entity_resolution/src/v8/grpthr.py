"""Separate (S2,S3) thresholds for S1 entities whose normalised name is shared vs unique; val split."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
probs = sys.argv[1] if len(sys.argv) > 1 else "work7/val_scored_all_v7.parquet"
cfg = Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1, pool = load_prepared(cfg, "train")
k = s1["country"].astype(str)
shared = (s1.groupby([k, s1["name_key"]])["name_key"].transform("size").to_numpy() > 1)
d = pd.read_parquet(probs)[["q", "p", "src", "prob"]]
u = np.random.default_rng(cfg.seed).random(len(s1)); val_q = np.flatnonzero(u < cfg.val_s1_frac); vset = set(val_q.tolist())
gt = load_gt(cfg); sp = pd.Series(np.arange(len(s1)), index=s1.entity_id.values); pp = pd.Series(np.arange(len(pool)), index=pool.entity_id.values)
gt = gt[gt.entity_id.isin(pp.index)]; gq, gp = sp[gt.source1_entity_id].values, pp[gt.entity_id].values
true_qp = {}
for q, p in zip(gq, gp):
    if q in vset: true_qp.setdefault(q, set()).add(p)
b = assign_best(d.q.to_numpy(), d.p.to_numpy(), d.prob.to_numpy()).merge(d[["q", "p", "src"]], on=["q", "p"])
b = b[b.q.isin(vset)].reset_index(drop=True)
sh = shared[b.q.to_numpy()]; s2 = b.src.to_numpy() == 2; pr = b.prob.to_numpy()
fs = {}
for q in val_q:
    fs[q] = None
def macro(mask, qs):
    pred = {}
    for q, p in zip(b.q.to_numpy()[mask], b.p.to_numpy()[mask]): pred.setdefault(q, set()).add(p)
    return np.mean([f05_entity(pred.get(q, set()), true_qp.get(q, set())) for q in qs])
qsh = val_q[shared[val_q]]; qun = val_q[~shared[val_q]]
grid = np.round(np.arange(0.40, 0.951, 0.05), 3)
best = {}
for name, grp, qs in (("shared", sh, qsh), ("unique", ~sh, qun)):
    res = {}
    for t2 in grid:
        for t3 in grid:
            if abs(t2 - t3) > 0.10: continue
            m = grp & (pr >= np.where(s2, t2, t3))
            res[(t2, t3)] = macro(m, qs)
    kk = max(res, key=res.get); best[name] = kk
    base = res.get((0.75, 0.7), res.get((0.75, 0.75)))
    print(f"[grp] {name}: {len(qs):,} entities; best thr {kk} F0.5 {res[kk]:.5f} (at 0.75/0.70: {base:.5f})", flush=True)
thr = np.where(sh, np.where(s2, best["shared"][0], best["shared"][1]), np.where(s2, best["unique"][0], best["unique"][1]))
print(f"[grp] combined per-group: F0.5 {macro(pr >= thr, val_q):.5f} | global 0.75/0.70: {macro(pr >= np.where(s2, 0.75, 0.70), val_q):.5f}", flush=True)
print("GRP_DONE", flush=True)
