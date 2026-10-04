"""Measure France with the labelled synthetic French world (dataset_syn + syn_gt.tsv).

France is 15% of the test set and has no training labels, so it is invisible to the normal
validation split. The synthetic world reuses the real French test S1 records, generates their
copies with the observed French noise, and fills the pool with real French test records the model
is confident match nothing - so crowding, names and decoy density are the country's real ones.
"""
import argparse, csv, json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared
from evaluate import f05_entity
from model import assign_best
from stage3 import _stage3_matrix

def read(p):
    return pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False)

ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work_syn")
ap.add_argument("--models", default="work7")
ap.add_argument("--ce", default="work_syn/syn_ce_ens.parquet")
ap.add_argument("--tag", default="v7")
a = ap.parse_args()
t = time.time()
cfg = Config(data_dir="dataset_syn", work_dir=a.work_dir, n_jobs=31)
s1, pool = load_prepared(cfg, "test")
meta = pd.read_parquet(cfg.path("s4_test_meta.parquet"))
X = np.load(cfg.path("s3_test_X.npy"), mmap_mode="r")
X3 = _stage3_matrix(np.asarray(X), meta, a.ce); del X
print(f"[fr] X3 {X3.shape}; {time.time()-t:.0f}s", flush=True)
ms = [lgb.Booster(model_file=os.path.join(a.models, f"model_s4_{i}.txt")) for i in range(6)]
prob = np.mean([m.predict(X3, num_threads=31) for m in ms], axis=0).astype(np.float32); del X3
pd.DataFrame({"q": meta["q"].values, "p": meta["p"].values, "prob": prob}).to_parquet(
    cfg.path(f"syn_scored_{a.tag}.parquet"), index=False)
gt = read(os.path.join("dataset_syn", "syn_gt.tsv"))
sp = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
pp = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
gt["lst"] = gt["matched_entity_ids"].str.split(",")
true_qp = {}
for eid, lst in zip(gt["source1_entity_id"].values, gt["lst"].values):
    q = sp.get(eid)
    if q is None: continue
    true_qp[q] = {pp[x] for x in lst if x and x in pp.index}
qs = np.arange(len(s1))
best = assign_best(meta["q"].values, meta["p"].values, prob).merge(meta[["q","p","src"]], on=["q","p"], how="left")
def ev(sel):
    pred = {}
    for q, p in zip(sel["q"].to_numpy(), sel["p"].to_numpy()): pred.setdefault(q, set()).add(p)
    f=tp=npd=ntr=0.0
    for q in qs:
        pr, tr = pred.get(q, set()), true_qp.get(q, set())
        f += f05_entity(pr, tr); tp += len(pr & tr); npd += len(pr); ntr += len(tr)
    return f/len(qs), tp/max(1.,npd), tp/max(1.,ntr)
rep={}
for t2 in np.round(np.arange(0.40,0.951,0.05),3):
    for t3 in (t2,):
        thr = np.where(best["src"].to_numpy()==2, t2, t3)
        r = ev(best[best["prob"].to_numpy()>=thr]); rep[f"thr{t2}"]=list(r)
        print(f"[fr] {a.tag} thr {t2}: F0.5 {r[0]:.5f} P {r[1]:.4f} R {r[2]:.4f}", flush=True)
for (t2,t3) in ((0.75,0.70),(0.70,0.65),(0.6,0.55),(0.5,0.45)):
    thr = np.where(best["src"].to_numpy()==2, t2, t3)
    r = ev(best[best["prob"].to_numpy()>=thr]); rep[f"src{t2}_{t3}"]=list(r)
    print(f"[fr] {a.tag} src({t2},{t3}): F0.5 {r[0]:.5f} P {r[1]:.4f} R {r[2]:.4f}", flush=True)
nt=sum(len(v) for v in true_qp.values()); incand=0
ck=set(zip(meta["q"].tolist(), meta["p"].tolist()))
for q,v in true_qp.items():
    for p in v: incand += (q,p) in ck
print(f"[fr] entities {len(qs):,} true pairs {nt:,} in candidates {incand/max(1,nt):.4f}; {time.time()-t:.0f}s", flush=True)
json.dump(rep, open(f"v8/fr_report_{a.tag}.json","w"), indent=1)
print("FR_DONE", flush=True)
