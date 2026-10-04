"""Rule A on validation: drop selected pairs whose pool record has an empty address and whose S1
entity's normalised name is shared by another S1 entity of the same country."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
cfg = Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1, pool = load_prepared(cfg, "train")
k = s1["country"].astype(str)
shared = s1.groupby([k, s1["name_key"]])["name_key"].transform("size").to_numpy() > 1
d = pd.read_parquet("work7/val_scored_all_v7.parquet")[["q", "p", "src", "prob"]]
u = np.random.default_rng(cfg.seed).random(len(s1)); val_q = np.flatnonzero(u < cfg.val_s1_frac); vset = set(val_q.tolist())
gt = load_gt(cfg); sp = pd.Series(np.arange(len(s1)), index=s1.entity_id.values); pp = pd.Series(np.arange(len(pool)), index=pool.entity_id.values)
gt = gt[gt.entity_id.isin(pp.index)]; gq, gp = sp[gt.source1_entity_id].values, pp[gt.entity_id].values
true_qp = {}
for q, p in zip(gq, gp):
    if q in vset: true_qp.setdefault(q, set()).add(p)
b = assign_best(d.q.to_numpy(), d.p.to_numpy(), d.prob.to_numpy()).merge(d[["q", "p", "src"]], on=["q", "p"])
b = b[b.q.isin(vset)]
sel = b[b.prob.to_numpy() >= np.where(b.src.to_numpy() == 2, 0.75, 0.70)].reset_index(drop=True)
def ev(s):
    pred = {}
    for q, p in zip(s.q.to_numpy(), s.p.to_numpy()): pred.setdefault(q, set()).add(p)
    f = tp = npd = ntr = 0.0
    for q in val_q:
        x, y = pred.get(q, set()), true_qp.get(q, set()); f += f05_entity(x, y); tp += len(x & y); npd += len(x); ntr += len(y)
    return f / len(val_q), tp / max(1, npd), tp / max(1, ntr)
print("[ruleA] baseline F0.5 %.5f P %.4f R %.4f" % ev(sel), flush=True)
emp = pool.addr_empty.values[sel.p.to_numpy()] == 1; sh = shared[sel.q.to_numpy()]
tk = set(zip(gq.tolist(), gp.tolist()))
for name, m in (("A: empty addr & name shared", emp & sh), ("empty addr (any)", emp), ("A with prob<0.95", emp & sh & (sel.prob.to_numpy() < 0.95)), ("A with prob<0.9", emp & sh & (sel.prob.to_numpy() < 0.9))):
    y = np.array([(a, c) in tk for a, c in zip(sel.q.to_numpy()[m], sel.p.to_numpy()[m])])
    r = ev(sel[~m])
    print(f"[ruleA] drop {name}: {int(m.sum()):,} pairs, {y.mean():.3f} correct -> F0.5 {r[0]:.5f} P {r[1]:.4f} R {r[2]:.4f}", flush=True)
print("RULEA_DONE", flush=True)
