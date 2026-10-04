"""Apply the stage-5 corrector to the labelled synthetic French world and re-measure France."""
import argparse, csv, json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "code/src"); sys.path.insert(0, "v8")
from config import Config
from data import load_prepared
from evaluate import f05_entity
from model import assign_best
from stage5 import FEATS, SUP, _ctx, CMAP

def read(p):
    return pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False)

ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work_syn")
ap.add_argument("--models", default="work7")
ap.add_argument("--ce", default="work_syn/syn_ce_ens.parquet")
ap.add_argument("--ce8", default="work_syn/v8_syn_ce8.parquet")
ap.add_argument("--prob", default="work_syn/syn_scored_v7.parquet")
ap.add_argument("--tag", default="v8")
a = ap.parse_args()
t = time.time()
cfg = Config(data_dir="dataset_syn", work_dir=a.work_dir, n_jobs=31)
s1, pool = load_prepared(cfg, "test")
meta = pd.read_parquet(cfg.path("s4_test_meta.parquet"), columns=["q","p","src","prob2"]+SUP)
pr = pd.read_parquet(a.prob)
d = meta.merge(pr, on=["q","p"], how="left")
full = d[["q","p","src","prob"]].copy()
ce = pd.read_parquet(a.ce); c8 = pd.read_parquet(a.ce8)
band = c8[["q","p"]]
d = band.merge(d, on=["q","p"], how="inner").merge(ce, on=["q","p"], how="left").merge(c8, on=["q","p"], how="left")
d["ctry"] = np.int8(CMAP["France"])
d["addr_empty"] = pool["addr_empty"].values[d["p"].to_numpy()].astype(np.int8)
d["d_ce8_ce"] = (d["ce8"]-d["ce"]).astype(np.float32)
d["d_ce8_prob"] = (d["ce8"]-d["prob"]).astype(np.float32)
d["d_ce_prob"] = (d["ce"]-d["prob"]).astype(np.float32)
d["n_q"] = d.groupby("q")["p"].transform("size").to_numpy(np.int32)
d["n_p"] = d.groupby("p")["q"].transform("size").to_numpy(np.int32)
d["n_q_src"] = d.groupby(["q","src"])["p"].transform("size").to_numpy(np.int32)
for k,v in _ctx(d).items(): d[k]=v
ms = [lgb.Booster(model_file=os.path.join(a.models, f"model_s5_{i}.txt")) for i in (0,1)]
pc = np.mean([m.predict(d[FEATS].to_numpy(np.float32), num_threads=31) for m in ms], axis=0).astype(np.float32)
fu = full.merge(pd.DataFrame({"q":d["q"].to_numpy(),"p":d["p"].to_numpy(),"pc":pc}), on=["q","p"], how="left")
pf = np.where(np.isnan(fu["pc"].to_numpy()), fu["prob"].to_numpy(), fu["pc"].to_numpy()).astype(np.float32)
print(f"[fr5] band {len(d):,} of {len(full):,}; {time.time()-t:.0f}s", flush=True)
gt = read(os.path.join("dataset_syn","syn_gt.tsv"))
sp = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
pp = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
gt["lst"]=gt["matched_entity_ids"].str.split(",")
true_qp={}
for eid,lst in zip(gt["source1_entity_id"].values, gt["lst"].values):
    q=sp.get(eid)
    if q is None: continue
    true_qp[q]={pp[x] for x in lst if x and x in pp.index}
qs=np.arange(len(s1))
def ev(sel):
    pred={}
    for q,p in zip(sel["q"].to_numpy(), sel["p"].to_numpy()): pred.setdefault(q,set()).add(p)
    f=tp=npd=ntr=0.0
    for q in qs:
        a_,b_=pred.get(q,set()), true_qp.get(q,set())
        f+=f05_entity(a_,b_); tp+=len(a_&b_); npd+=len(a_); ntr+=len(b_)
    return f/len(qs), tp/max(1.,npd), tp/max(1.,ntr)
rep={}
for label,pv in (("v7", fu["prob"].to_numpy()), ("v8", pf)):
    best = assign_best(fu["q"].to_numpy(), fu["p"].to_numpy(), pv).merge(fu[["q","p","src"]], on=["q","p"], how="left")
    bb=-1;bk=None
    for t2 in np.round(np.arange(0.15,0.951,0.05),3):
        for t3 in np.round(np.arange(0.15,0.951,0.05),3):
            if abs(t2-t3)>0.15: continue
            thr=np.where(best["src"].to_numpy()==2,t2,t3)
            r=ev(best[best["prob"].to_numpy()>=thr])
            if r[0]>bb: bb,bk,br=r[0],(float(t2),float(t3)),r
    rep[label]={"best_thr":list(bk),"f05":br[0],"P":br[1],"R":br[2]}
    thr=np.where(best["src"].to_numpy()==2,0.75,0.70)
    r0=ev(best[best["prob"].to_numpy()>=thr]); rep[label]["at_v7_thr"]=list(r0)
    print(f"[fr5] {label}: best {bk} F0.5 {br[0]:.5f} P {br[1]:.4f} R {br[2]:.4f} | at(0.75,0.70) F0.5 {r0[0]:.5f} P {r0[1]:.4f} R {r0[2]:.4f}", flush=True)
json.dump(rep, open(f"v8/fr_report_{a.tag}.json","w"), indent=1)
print("FR5_DONE", flush=True)
