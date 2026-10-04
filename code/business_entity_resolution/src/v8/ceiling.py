import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
t=time.time()
cfg = Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1, pool = load_prepared(cfg, "train")
d = pd.read_parquet(cfg.path("val_scored_all_v7.parquet"))
u = np.random.default_rng(cfg.seed).random(len(s1)); val_q = np.flatnonzero(u < cfg.val_s1_frac); vset=set(val_q.tolist())
gt = load_gt(cfg)
sp = pd.Series(np.arange(len(s1)), index=s1.entity_id.values); pp = pd.Series(np.arange(len(pool)), index=pool.entity_id.values)
gt = gt[gt.entity_id.isin(pp.index)]
gq, gp = sp[gt.source1_entity_id].values, pp[gt.entity_id].values
true_qp={}
for q,p in zip(gq,gp):
    if q in vset: true_qp.setdefault(q,set()).add(p)
dv = d[d["q"].isin(vset)]
# ORACLE 1: predict exactly the true pairs present in the candidate set
cand_true={}
tq = dv[dv["y"].to_numpy()]
for q,p in zip(tq["q"].values, tq["p"].values): cand_true.setdefault(q,set()).add(p)
f=np.mean([f05_entity(cand_true.get(q,set()), true_qp.get(q,set())) for q in val_q])
print(f"[ceil] oracle on blocked candidates: macro F0.5 {f:.5f}", flush=True)
# ORACLE 2: same but also respecting one-owner assignment (a true pair whose p was claimed by
# another entity with a HIGHER true label cannot happen - labels are unique - so identical)
# ORACLE 3: best possible with the CURRENT ranking = sweep a per-entity oracle cut on v7 prob
best = assign_best(d["q"].values, d["p"].values, d["prob"].values).merge(d[["q","p","src","y"]],on=["q","p"],how="left")
bv = best[best["q"].isin(vset)]
g = bv.sort_values(["q","prob"],ascending=[True,False])
# per entity, choose the j (0..n) maximising F0.5 with hindsight -> ranking-limited ceiling
rows=[]; 
arr_q=g["q"].to_numpy(); arr_y=g["y"].to_numpy().astype(bool)
bounds=np.flatnonzero(np.diff(arr_q))+1
tot=0.0
for grp in np.split(np.arange(len(arr_q)), bounds):
    q=arr_q[grp[0]]; tr=true_qp.get(q,set()); nt=len(tr)
    ys=arr_y[grp]; cs=np.cumsum(ys); bestf=1.0 if nt==0 else 0.0
    for j in range(1,len(grp)+1):
        tp=cs[j-1]
        if tp==0 or nt==0: fv=0.0
        else:
            p_=tp/j; r_=tp/nt; fv=1.25*p_*r_/(0.25*p_+r_)
        if fv>bestf: bestf=fv
    tot+=bestf
print(f"[ceil] oracle cut on v7 ranking (per-entity hindsight j): macro F0.5 {tot/len(val_q):.5f}", flush=True)
# how many val entities have NO true pair in candidates but do have true pairs
miss_all=sum(1 for q in val_q if true_qp.get(q) and not cand_true.get(q))
print(f"[ceil] val entities {len(val_q):,}; with true pairs {sum(1 for q in val_q if true_qp.get(q)):,}; "
      f"all-true-blocked-out {miss_all:,}; {time.time()-t:.0f}s", flush=True)
print("CEIL_DONE", flush=True)
