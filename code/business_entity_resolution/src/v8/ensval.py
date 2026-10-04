"""Probability-level ensemble of independently trained runs, measured on the held-out split."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0,"code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
cfg=Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1,pool=load_prepared(cfg,"train")
u=np.random.default_rng(cfg.seed).random(len(s1)); val_q=np.flatnonzero(u<cfg.val_s1_frac); vset=set(val_q.tolist())
gt=load_gt(cfg); sp=pd.Series(np.arange(len(s1)),index=s1.entity_id.values); pp=pd.Series(np.arange(len(pool)),index=pool.entity_id.values)
gt=gt[gt.entity_id.isin(pp.index)]; gq,gp=sp[gt.source1_entity_id].values, pp[gt.entity_id].values
true_qp={}
for q,p in zip(gq,gp):
    if q in vset: true_qp.setdefault(q,set()).add(p)
a=pd.read_parquet("work7/val_scored_all_v7.parquet")[["q","p","src","prob"]].rename(columns={"prob":"p7"})
cands=[]
for name,path in [("v6","work6/val_scored_final.parquet"),("v6all","work6/val_scored_all.parquet"),("w","work/val_scored_all.parquet")]:
    if os.path.exists(path):
        b=pd.read_parquet(path); print(name, path, b.shape, list(b.columns)[:6], flush=True); cands.append((name,b))
if not cands: print("no other val prob files"); sys.exit()
name,b=cands[0]
b=b[["q","p","prob"]].rename(columns={"prob":"pb"})
m=a.merge(b,on=["q","p"],how="left"); print("overlap", m.pb.notna().mean(), flush=True)
m["pb"]=m["pb"].fillna(m["p7"])
def ev(sel):
    pred={}
    for q,p in zip(sel["q"].to_numpy(), sel["p"].to_numpy()): pred.setdefault(q,set()).add(p)
    f=tp=npd=ntr=0.0
    for q in val_q:
        x,y=pred.get(q,set()),true_qp.get(q,set()); f+=f05_entity(x,y); tp+=len(x&y); npd+=len(x); ntr+=len(y)
    return f/len(val_q), tp/max(1.,npd), tp/max(1.,ntr)
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    pv=(w*m["pb"].to_numpy()+(1-w)*m["p7"].to_numpy()).astype(np.float32)
    best=assign_best(m["q"].to_numpy(), m["p"].to_numpy(), pv).merge(m[["q","p","src"]],on=["q","p"],how="left")
    bv=best[best["q"].isin(vset)]
    out=[]
    for t2,t3 in ((0.75,0.75),(0.75,0.70),(0.70,0.70),(0.80,0.75),(0.70,0.65)):
        thr=np.where(bv["src"].to_numpy()==2,t2,t3); r=ev(bv[bv["prob"].to_numpy()>=thr]); out.append((t2,t3,r))
    t2,t3,r=max(out,key=lambda x:x[2][0])
    print(f"[ens] w_{name}={w}: best thr ({t2},{t3}) F0.5 {r[0]:.5f} P {r[1]:.4f} R {r[2]:.4f}", flush=True)
print("ENS_DONE", flush=True)
