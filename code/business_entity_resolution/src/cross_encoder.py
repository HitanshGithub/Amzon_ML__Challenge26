"""Transformer cross-encoder for candidate pairs (v3).

A multilingual encoder (default intfloat/multilingual-e5-base, MIT licence, 278M params) reads
both records as raw text - "name | address" - and predicts P(same business). Raw text is used on
purpose so the pretrained multilingual knowledge sees accents, Indic scripts and French words.

  train : pairs from training S1 entities outside the validation split (same split and seed as
          run_pipeline.stage_train): all true pairs + the hardest blocking negatives per entity.
  score : scores any (q, p) pair table of a split and writes a parquet with column 'ce'.

Only the provided training data is used; the pretrained weights come from the Hugging Face hub.
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import load_gt, raw_paths, read_tsv
from config import Config


def record_texts(cfg, split):
    """Raw 'name | address' strings, row-aligned with the prepared parquet files."""
    p = raw_paths(cfg, split)
    s1 = read_tsv(p["source1"])
    pool = pd.concat([read_tsv(p["source2"]), read_tsv(p["source3"])], ignore_index=True)
    fmt = lambda df: (df["business_name"].str.strip() + " | " + df["business_address"].str.strip()).values
    return fmt(s1), fmt(pool), s1["entity_id"].values, pool["entity_id"].values


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


class LengthBucketSampler(torch.utils.data.Sampler):
    """Random batches of similar-length pairs: shuffle, cut into chunks of 100 batches, sort each
    chunk by length, split into batches, shuffle batch order. Cuts padding (and GPU time) a lot."""

    def __init__(self, lengths, bs, seed=0):
        self.lengths, self.bs, self.seed, self.epoch = np.asarray(lengths), bs, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        chunk = self.bs * 100
        batches = []
        for s in range(0, len(idx), chunk):
            c = idx[s:s + chunk]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[i:i + self.bs] for i in range(0, len(c) - self.bs + 1, self.bs)]
        for i in rng.permutation(len(batches)):
            yield batches[i].tolist()

    def __len__(self):
        return len(self.lengths) // self.bs


def make_collate(tok, max_len):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return collate


def build_training_pairs(cfg, n_s1, neg_hard, neg_rand, cands_path, seed=0):
    s1 = pd.read_parquet(cfg.path("train_s1.parquet"), columns=["entity_id"])
    pool = pd.read_parquet(cfg.path("train_pool.parquet"), columns=["entity_id"])
    u = np.random.default_rng(cfg.seed).random(len(s1))               # identical to stage_train
    rng = np.random.default_rng(seed)
    train_q = np.flatnonzero(u >= cfg.val_s1_frac)
    train_q = rng.choice(train_q, size=min(n_s1, len(train_q)), replace=False)
    val_q = np.flatnonzero(u < cfg.val_s1_frac)

    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index)]
    gq, gp = s1_pos[gt["source1_entity_id"]].values, p_pos[gt["entity_id"]].values
    true_keys = gq.astype(np.int64) * len(pool) + gp

    c = pd.read_parquet(cands_path, columns=["q", "p", "comb"])
    c = c[np.isin(c["q"].values, train_q)]
    c["y"] = np.isin(c["q"].values.astype(np.int64) * len(pool) + c["p"].values, true_keys)
    neg = c[~c["y"]].sort_values(["q", "comb"], ascending=[True, False])
    rank = neg.groupby("q").cumcount()
    hard = neg[rank < neg_hard]
    rest = neg[rank >= neg_hard]
    rand = rest.sample(frac=1.0, random_state=seed).groupby("q").head(neg_rand)
    m = np.isin(gq, train_q)
    pos = pd.DataFrame({"q": gq[m], "p": gp[m]})
    pairs = pd.concat([pos.assign(y=1.0), hard[["q", "p"]].assign(y=0.0), rand[["q", "p"]].assign(y=0.0)],
                      ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pairs, val_q


@torch.no_grad()
def predict(model, tok, a, b, max_len, bs, device):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])          # length-sorted = less padding
    out = np.empty(len(a), dtype=np.float32)
    collate = make_collate(tok, max_len)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = collate([(a[i], b[i], 0.0) for i in idx])
        enc.pop("labels")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**enc).logits.float().squeeze(-1)
        out[idx] = torch.sigmoid(logits).cpu().numpy()
    return out


def cmd_train(a):
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)
    t = time.time()
    ta, tp, _, _ = record_texts(cfg, "train")
    if a.pairs_file:
        # v4: labelled (q, p, y) list built by stage4.py pairs - stage-2 hard negatives - plus
        # optional pseudo-labelled pairs from another split (France test records)
        pairs = pd.read_parquet(a.pairs_file)
        if a.n_s1 and len(pairs) > a.n_s1:
            pairs = pairs.sample(n=a.n_s1, random_state=4)
        A, B, y = ta[pairs["q"].values], tp[pairs["p"].values], pairs["y"].values.astype(np.float32)
        if a.extra_pairs:
            ex = pd.concat([pd.read_parquet(p) for p in a.extra_pairs.split(",")], ignore_index=True)
            ea, ep, _, _ = record_texts(cfg, a.extra_split)
            A = np.concatenate([A, ea[ex["q"].values]])
            B = np.concatenate([B, ep[ex["p"].values]])
            y = np.concatenate([y, ex["y"].values.astype(np.float32)])
            print(f"[ce] + {len(ex):,} extra pairs from split {a.extra_split!r} ({ex['y'].mean():.3f} positive)", flush=True)
        if a.text_pairs:
            # labelled raw-text pairs (e.g. synth_fr.py output for a country without labels)
            tx = pd.read_parquet(a.text_pairs)
            A = np.concatenate([A, tx["a"].values]); B = np.concatenate([B, tx["b"].values])
            y = np.concatenate([y, tx["y"].values.astype(np.float32)])
            print(f"[ce] + {len(tx):,} text pairs from {a.text_pairs} ({tx['y'].mean():.3f} positive)", flush=True)
        perm = np.random.default_rng(0).permutation(len(A))
        A, B, y = A[perm], B[perm], y[perm]
        u = np.random.default_rng(cfg.seed).random(len(ta))
        val_q = np.flatnonzero(u < cfg.val_s1_frac)
    else:
        pairs, val_q = build_training_pairs(cfg, a.n_s1, a.neg_hard, a.neg_rand, a.cands)
        A, B, y = ta[pairs["q"].values], tp[pairs["p"].values], pairs["y"].values.astype(np.float32)
    print(f"[ce] {len(A):,} training pairs ({y.mean():.3f} positive) built in {time.time() - t:.0f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(device)
    a.max_len = min(a.max_len, model.config.max_position_embeddings - 2)
    lengths = np.fromiter((len(x) + len(z) for x, z in zip(A, B)), dtype=np.int32, count=len(A))
    dl = DataLoader(Pairs(A, B, y), batch_sampler=LengthBucketSampler(lengths, a.bs), num_workers=a.workers,
                    collate_fn=make_collate(tok, a.max_len), pin_memory=True)
    steps = len(dl) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t = 0, time.time()
    model.train()
    for ep in range(a.epochs):
        for enc in dl:
            labels = enc.pop("labels").to(device, non_blocking=True)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**enc).logits.float().squeeze(-1)
            loss = lossf(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0 or step == steps:
                el = time.time() - t
                print(f"[ce] step {step}/{steps} loss {loss.item():.4f} {step * a.bs / el:.0f} pairs/s "
                      f"eta {(steps - step) * el / step / 60:.0f} min", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)

    # held-out check on validation entities (never used for training): all their blocking candidates
    c = pd.read_parquet(a.cands, columns=["q", "p", "comb"])
    vq = np.random.default_rng(1).choice(val_q, size=min(a.n_val, len(val_q)), replace=False)
    c = c[np.isin(c["q"].values, vq)].reset_index(drop=True)
    s1 = pd.read_parquet(cfg.path("train_s1.parquet"), columns=["entity_id"])
    pool = pd.read_parquet(cfg.path("train_pool.parquet"), columns=["entity_id"])
    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index)]
    tk = s1_pos[gt["source1_entity_id"]].values.astype(np.int64) * len(pool) + p_pos[gt["entity_id"]].values
    yv = np.isin(c["q"].values.astype(np.int64) * len(pool) + c["p"].values, tk)
    c["ce"] = predict(model, tok, ta[c["q"].values], tp[c["p"].values], a.max_len, a.bs * 4, device)
    eps = 1e-6
    ll = -np.mean(yv * np.log(c["ce"] + eps) + (1 - yv) * np.log(1 - c["ce"] + eps))
    top1 = c.sort_values("ce", ascending=False).drop_duplicates("q")
    print(f"[ce] validation: {len(c):,} pairs of {len(vq):,} entities, logloss {ll:.5f}, "
          f"precision@0.5 {yv[c['ce'] > 0.5].mean():.4f}, recall@0.5 {(c['ce'][yv] > 0.5).mean():.4f}, "
          f"top-1 precision {np.isin(top1['q'].values.astype(np.int64) * len(pool) + top1['p'].values, tk).mean():.4f}", flush=True)
    c.assign(y=yv)[["q", "p", "ce", "y"]].to_parquet(os.path.join(a.out, "val_scores.parquet"), index=False)


def cmd_score(a):
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.out)
    model = AutoModelForSequenceClassification.from_pretrained(a.out).to(device)
    a.max_len = min(a.max_len, model.config.max_position_embeddings - 2)
    c = pd.read_parquet(a.pairs, columns=["q", "p"])
    ta, tp, _, _ = record_texts(cfg, a.split)
    t = time.time()
    ce = np.empty(len(c), dtype=np.float32)
    step = 2_000_000
    for s in range(0, len(c), step):
        q, p = c["q"].values[s:s + step], c["p"].values[s:s + step]
        ce[s:s + step] = predict(model, tok, ta[q], tp[p], a.max_len, a.bs, device)
        print(f"[ce-score] {min(s + step, len(c)):,}/{len(c):,} pairs, {(s + len(q)) / (time.time() - t):.0f} pairs/s", flush=True)
    c["ce"] = ce
    c.to_parquet(a.dest, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--cands", default="work/train_cands.parquet")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", default="work/ce_model")
    ap.add_argument("--n-s1", type=int, default=300_000)
    ap.add_argument("--neg-hard", type=int, default=6)
    ap.add_argument("--neg-rand", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=20_000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs-file", default="", help="labelled (q,p,y) training pairs on the train split")
    ap.add_argument("--extra-pairs", default="", help="comma list of labelled (q,p,y) pair files on --extra-split")
    ap.add_argument("--extra-split", default="test")
    ap.add_argument("--text-pairs", default="", help="parquet with raw-text columns a, b, y to add to training")
    ap.add_argument("--pairs", default="work/test_cands.parquet")
    ap.add_argument("--dest", default="work/test_ce.parquet")
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_score(a)


if __name__ == "__main__":
    main()
