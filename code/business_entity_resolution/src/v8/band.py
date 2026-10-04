"""Pairs worth a ce8 score: the band where the v7 cross-encoder is not already sure, plus every
French pair (France has no training labels, so that is where the new model must earn its keep)."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "code/src")
from config import Config
from data import raw_paths, read_tsv
cfg = Config(data_dir="dataset", work_dir="work")
LO, HI = 0.0005, 0.9995
for split, cef in (("train", "work/v7_train_ce_fr.parquet"), ("test", "work/v7_test_ce_fr.parquet")):
    ce = pd.read_parquet(cef)
    c = ce["ce"].to_numpy()
    m = (c >= LO) & (c <= HI)
    if split == "test":
        s1 = read_tsv(raw_paths(cfg, split)["source1"])
        fr = (s1["country"].values == "France")
        m = m | fr[ce["q"].to_numpy()]
    # plus the top-2 by ce inside each entity, so within-entity ce8 ranks are meaningful
    rk = ce.groupby("q")["ce"].rank(ascending=False, method="first").to_numpy()
    m = m | (rk <= 2)
    out = ce.loc[m, ["q", "p"]].reset_index(drop=True)
    out.to_parquet(f"work/v8_{split}_band.parquet", index=False)
    print(f"[band] {split}: {len(out):,} of {len(ce):,} ({m.mean():.3f})", flush=True)
print("BAND_DONE", flush=True)
