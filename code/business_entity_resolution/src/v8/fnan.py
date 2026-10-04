"""Which below-threshold true pairs could be rescued at a precision F0.5 can afford?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"code/src")
from config import Config
from data import load_prepared, load_gt
cfg=Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1,pool=load_prepared(cfg,"train")
c=pd.read_parquet(cfg.path("v8_rescue_cand.parquet"))
gt=load_gt(cfg)
sp=pd.Series(np.arange(len(s1)),index=s1.entity_id.values); pp=pd.Series(np.arange(len(pool)),index=pool.entity_id.values)
gt=gt[gt.entity_id.isin(pp.index)]
tk=set(zip(sp[gt.source1_entity_id].values.tolist(), pp[gt.entity_id].values.tolist()))
c["y"]=[ (q,p) in tk for q,p in zip(c.q.values,c.p.values)]
c["ctry"]=s1.country.values[c.q.values]
for col in ["addr_empty","non_ascii","has_alias","is_domain"]:
    if col in pool.columns: c["b_"+col]=pool[col].values[c.p.values]
print(f"rescue candidates {len(c):,}, hit-rate {c.y.mean():.3f}")
print("\nby prob bucket:")
c["pb"]=pd.cut(c.prob,[0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.76])
print(c.groupby("pb", observed=True).agg(n=("y","size"), hit=("y","mean")).round(3).to_string())
print("\nby sib/sib_a:")
c["sb"]=pd.cut(c.sib,[0,0.6,0.8,0.9,0.95,1.0])
c["sa"]=pd.cut(c.sib_a,[-0.01,0.001,0.8,0.9,0.95,1.0])
print(c.pivot_table(index="sb",columns="sa",values="y",aggfunc=["size","mean"],observed=True).round(3).to_string())
print("\nby traits (hit-rate):")
for col in [x for x in c.columns if x.startswith("b_")]+["ctry","src","need"]:
    print(" ", col, c.groupby(col, observed=True).agg(n=("y","size"),hit=("y","mean")).round(3).to_dict())
print("\nhigh-precision pockets (need hit>0.667 to help F0.5 on a 2-true entity):")
for name,m in [("sib_a==1 & sib>=0.9", (c.sib_a>=0.999)&(c.sib>=0.9)),
               ("sib_a==1 & prob>=0.3", (c.sib_a>=0.999)&(c.prob>=0.3)),
               ("prob>=0.5", c.prob>=0.5), ("prob>=0.6", c.prob>=0.6),
               ("addr_empty=0 & sib_a>=0.95", (c.get("b_addr_empty",pd.Series(0,index=c.index))==0)&(c.sib_a>=0.95)),
               ("rank1 & prob>=0.4", (c.rk==1)&(c.prob>=0.4)),
               ("need & rank1 & prob>=0.5", c.need&(c.rk==1)&(c.prob>=0.5))]:
    if m.sum(): print(f"  {name:34s} n={int(m.sum()):6d} hit={c.y[m].mean():.3f}")
