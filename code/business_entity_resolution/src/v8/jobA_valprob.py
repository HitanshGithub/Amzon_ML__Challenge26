"""V8 Job A: reproduce v7's out-of-fold probabilities for every train pair.
Saves work7/val_scored_all_v7.parquet  (q, p, src, prob, prob2, y)  -> the decision lab."""
import sys, os, time
sys.path.insert(0, "code/src")
import numpy as np, pandas as pd, lightgbm as lgb
from config import Config
from data import load_prepared
from stage3 import _stage3_matrix
from stage4 import _true_keys, _key

t0 = time.time()
cfg = Config(data_dir="dataset", work_dir="work7", n_jobs=31)
s1, pool = load_prepared(cfg, "train")
meta = pd.read_parquet(cfg.path("s4_train_meta.parquet"))
gq, gp, tk = _true_keys(cfg, s1, pool)
sub = meta["in_s3"].to_numpy()
print(f"[A] meta {len(meta):,} in_s3 {sub.sum():,} ({time.time()-t0:.0f}s)", flush=True)
m3 = meta[sub].reset_index(drop=True)
X = np.load(cfg.path("s3_train_X.npy"), mmap_mode="r")
X3 = _stage3_matrix(np.asarray(X[sub]), m3, cfg.path("s4_train_ce_ensfr.parquet"))
del X
print(f"[A] X3 {X3.shape} ({time.time()-t0:.0f}s)", flush=True)

fold, is_val = m3["fold"].to_numpy(), m3["is_val"].to_numpy()
prob3 = np.zeros(len(m3), np.float32)
for s in range(3):
    for f in (0, 1):
        m = lgb.Booster(model_file=cfg.path(f"model_s4_{2*s+f}.txt"))
        oof = np.flatnonzero((fold != f) & ~is_val)
        prob3[oof] += m.predict(X3[oof], num_threads=cfg.n_jobs).astype(np.float32) / 3.0
        vi = np.flatnonzero(is_val)
        prob3[vi] += m.predict(X3[vi], num_threads=cfg.n_jobs).astype(np.float32) / 6.0
        print(f"[A] seed{s} fold{f} done ({time.time()-t0:.0f}s)", flush=True)
del X3
prob_all = meta["prob2"].to_numpy().copy().astype(np.float32)
prob_all[np.flatnonzero(sub)] = prob3
y = np.isin(_key(meta["q"].values, meta["p"].values, len(pool)), tk)
out = pd.DataFrame({"q": meta["q"].values.astype(np.int32), "p": meta["p"].values.astype(np.int32),
                    "src": meta["src"].values, "prob": prob_all,
                    "prob2": meta["prob2"].to_numpy(np.float32), "y": y})
out.to_parquet(cfg.path("val_scored_all_v7.parquet"), index=False)
# also keep the cross-encoder score + support cols aligned for rule work
keep = ["q", "p", "sup_name_max", "sup_addr_max", "sup_prob_of_best", "sup_n_similar", "sup_weighted",
        "sup_other_src_name_max"]
meta[keep].to_parquet(cfg.path("v8_train_sup.parquet"), index=False)
print(f"[A] saved {len(out):,} rows, pos {y.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
print("JOBA_DONE", flush=True)
