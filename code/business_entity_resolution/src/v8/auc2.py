import sys, numpy as np, pandas as pd
sys.path.insert(0,"code/src")
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
d=pd.read_parquet("work7/val_scored_all_v7.parquet")[["q","p","prob","prob2","y"]]
b=pd.read_parquet("work/v8_train_band.parquet").merge(d,on=["q","p"],how="inner")
b=b.merge(pd.read_parquet("work7/s4_train_ce_ensfr.parquet"),on=["q","p"],how="left")
b=b.merge(pd.read_parquet("work/v8_train_ce8v2.parquet"),on=["q","p"],how="left")
s1=pd.read_parquet("work7/train_s1.parquet",columns=["entity_id"])
u=np.random.default_rng(42).random(len(s1)); isval=(u<0.10)[b.q.to_numpy()]
print("band rows",len(b),"val rows",int(isval.sum()),"pos",round(float(b.y.mean()),4), "ce8 nan",int(b.ce8.isna().sum()))
for name in ["prob","prob2","ce","ce8"]:
    v=b[name].to_numpy(); y=b.y.to_numpy().astype(int); m=isval & ~np.isnan(v)
    print(f"  {name:6s} AUC {roc_auc_score(y[m],v[m]):.5f}  AP {average_precision_score(y[m],v[m]):.5f}  logloss {log_loss(y[m],np.clip(v[m],1e-6,1-1e-6)):.5f}")
# where old ce is unsure, is ce8 decisive?
m=isval & (b.ce.to_numpy()>0.05)&(b.ce.to_numpy()<0.95)
y=b.y.to_numpy()[m].astype(int)
print("old-CE unsure rows",int(m.sum()),"pos",round(float(y.mean()),3))
for name in ["prob","ce","ce8"]:
    v=b[name].to_numpy()[m]; print(f"  {name:6s} AUC {roc_auc_score(y,v):.5f}  AP {average_precision_score(y,v):.5f}")
# disagreement: ce8 confident vs ce unsure
v8=b.ce8.to_numpy()[m]; print("  ce8>=0.9 among unsure:",int((v8>=0.9).sum()),"precision",round(float(y[v8>=0.9].mean()),3),"| ce8<=0.1:",int((v8<=0.1).sum()),"neg-precision",round(float(1-y[v8<=0.1].mean()),3))
# alignment sanity: correlation ce vs ce8 on confident old rows
mm=isval&(b.ce.to_numpy()>0.999); print("old ce>0.999 rows: mean ce8",round(float(b.ce8.to_numpy()[mm].mean()),3),"| old ce<0.001: mean ce8",round(float(b.ce8.to_numpy()[isval&(b.ce.to_numpy()<0.001)].mean()),3))
