"""Is the val->test gap just decoy density? Re-score v7 on val with test-like decoy density."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"code/src")
from config import Config
from data import load_prepared, load_gt
from evaluate import f05_entity
from model import assign_best
cfg=Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1,pool=load_prepared(cfg,"train")
d=pd.read_parquet(cfg.path("val_scored_all_v7.parquet"))
u=np.random.default_rng(cfg.seed).random(len(s1)); val_q=np.flatnonzero(u<cfg.val_s1_frac); vset=set(val_q.tolist())
gt=load_gt(cfg); sp=pd.Series(np.arange(len(s1)),index=s1.entity_id.values); pp=pd.Series(np.arange(len(pool)),index=pool.entity_id.values)
gt=gt[gt.entity_id.isin(pp.index)]
gq,gp=sp[gt.source1_entity_id].values, pp[gt.entity_id].values
true_qp={}
for q,p in zip(gq,gp):
    if q in vset: true_qp.setdefault(q,set()).add(p)
def run(drop):
    dq=np.zeros(len(s1),bool)
    if drop>0:
        sh=drop/(1-cfg.val_s1_frac); dq=(u>=cfg.val_s1_frac)&(u<cfg.val_s1_frac+sh*(1-cfg.val_s1_frac))
    keep=~dq[d["q"].to_numpy()]
    b=assign_best(d["q"].to_numpy()[keep], d["p"].to_numpy()[keep], d["prob"].to_numpy()[keep]).merge(d[["q","p","src"]],on=["q","p"],how="left")
    bv=b[b["q"].isin(vset)]
    out=[]
    for t2,t3 in ((0.75,0.70),(0.70,0.65),(0.65,0.60),(0.80,0.75)):
        thr=np.where(bv["src"].to_numpy()==2,t2,t3); sel=bv[bv["prob"].to_numpy()>=thr]
        pred={}
        for q,p in zip(sel["q"].to_numpy(), sel["p"].to_numpy()): pred.setdefault(q,set()).add(p)
        f=tp=npd=ntr=0.0
        for q in val_q:
            a_,b_=pred.get(q,set()),true_qp.get(q,set()); f+=f05_entity(a_,b_); tp+=len(a_&b_); npd+=len(a_); ntr+=len(b_)
        out.append((t2,t3,f/len(val_q),tp/max(1.,npd),tp/max(1.,ntr)))
    pool_per=len(pool)*(1-0)/max(1,int((~dq).sum()))
    print(f"[drop={drop}] kept S1 {int((~dq).sum()):,} pool/S1 {len(pool)/int((~dq).sum()):.2f}", flush=True)
    for t2,t3,f,P,R in out: print(f"    src({t2},{t3}): F0.5 {f:.5f} P {P:.4f} R {R:.4f}", flush=True)
for drop in (0.0, 0.187, 0.30):
    run(drop)
print("DROPVAL_DONE", flush=True)
