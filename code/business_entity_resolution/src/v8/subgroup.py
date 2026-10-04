"""Where is macro-F0.5 lost? Split S1 entities by whether their exact normalised name / address is
shared with another S1 entity of the same country (look-alike twins), on the val split (US/India)
and on the labelled synthetic French world."""
import sys, csv, os, numpy as np, pandas as pd
sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
def groups(s1):
    k = s1["country"].astype(str)
    nm = s1.groupby([k, s1["name_key"]])["name_key"].transform("size").to_numpy()
    ad = s1.groupby([k, s1["addr_norm"]])["addr_norm"].transform("size").to_numpy()
    return nm > 1, ad > 1
def report(tag, qs, pred, true_qp, s1, nm, ad):
    rows = []
    for q in qs:
        pr, tr = pred.get(q, set()), true_qp.get(q, set())
        rows.append((q, f05_entity(pr, tr), len(pr & tr), len(pr), len(tr)))
    r = pd.DataFrame(rows, columns=["q", "f", "tp", "np", "nt"])
    r["name_sh"] = nm[r.q.to_numpy()]; r["addr_sh"] = ad[r.q.to_numpy()]
    r["ctry"] = s1["country"].values[r.q.to_numpy()]
    tot_loss = (1 - r.f).sum()
    print(f"\n[{tag}] entities {len(r):,} macro F0.5 {r.f.mean():.5f}")
    for (c, n, a), g in r.groupby(["ctry", "name_sh", "addr_sh"]):
        P = g.tp.sum() / max(1, g.np.sum()); R = g.tp.sum() / max(1, g.nt.sum())
        print(f"  {c:7s} name_shared={int(n)} addr_shared={int(a)}: {len(g)/len(r):6.1%} of entities  F0.5 {g.f.mean():.4f}  "
              f"P {P:.4f} R {R:.4f}  share of loss {(1-g.f).sum()/tot_loss:6.1%}", flush=True)
# ---- val (US/India)
cfg = Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1, pool = load_prepared(cfg, "train"); nm, ad = groups(s1)
d = pd.read_parquet(cfg.path("val_scored_all_v7.parquet"))[["q", "p", "src", "prob"]]
u = np.random.default_rng(cfg.seed).random(len(s1)); val_q = np.flatnonzero(u < cfg.val_s1_frac); vset = set(val_q.tolist())
gt = load_gt(cfg); sp = pd.Series(np.arange(len(s1)), index=s1.entity_id.values); pp = pd.Series(np.arange(len(pool)), index=pool.entity_id.values)
gt = gt[gt.entity_id.isin(pp.index)]; gq, gp = sp[gt.source1_entity_id].values, pp[gt.entity_id].values
true_qp = {}
for q, p in zip(gq, gp):
    if q in vset: true_qp.setdefault(q, set()).add(p)
b = assign_best(d.q.to_numpy(), d.p.to_numpy(), d.prob.to_numpy()).merge(d[["q", "p", "src"]], on=["q", "p"])
b = b[b.q.isin(vset)]; sel = b[b.prob.to_numpy() >= np.where(b.src.to_numpy() == 2, 0.75, 0.70)]
pred = {}
for q, p in zip(sel.q.to_numpy(), sel.p.to_numpy()): pred.setdefault(q, set()).add(p)
report("val US/India, v7", val_q, pred, true_qp, s1, nm, ad)
del d, b
# ---- synthetic France
cfg2 = Config(data_dir="dataset_syn", work_dir="work_syn", n_jobs=31)
s1f, poolf = load_prepared(cfg2, "test"); nmf, adf = groups(s1f)
sc = pd.read_parquet("work_syn/syn_scored_v7.parquet")
meta = pd.read_parquet("work_syn/s4_test_meta.parquet", columns=["q", "p", "src"])
bf = assign_best(sc.q.to_numpy(), sc.p.to_numpy(), sc.prob.to_numpy()).merge(meta, on=["q", "p"])
self_ = bf[bf.prob.to_numpy() >= np.where(bf.src.to_numpy() == 2, 0.75, 0.70)]
predf = {}
for q, p in zip(self_.q.to_numpy(), self_.p.to_numpy()): predf.setdefault(q, set()).add(p)
g = pd.read_csv("dataset_syn/syn_gt.tsv", sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
spf = pd.Series(np.arange(len(s1f)), index=s1f.entity_id.values); ppf = pd.Series(np.arange(len(poolf)), index=poolf.entity_id.values)
tf = {}
for e, l in zip(g.source1_entity_id, g.matched_entity_ids):
    q = spf.get(e)
    if q is not None: tf[q] = {ppf[x] for x in l.split(",") if x and x in ppf.index}
report("synthetic France, v7", np.arange(len(s1f)), predf, tf, s1f, nmf, adf)
print("SUBGROUP_DONE", flush=True)
