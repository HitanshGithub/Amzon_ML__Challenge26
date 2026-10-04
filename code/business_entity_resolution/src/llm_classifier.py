"""LoRA-fine-tuned LLM pair classifier (Qwen2.5-7B-Instruct, Apache-2.0, 7.6B parameters).

Unlike the zero-shot judge (AUC 0.60 on uncertain pairs) this model is *trained* on the labelled
candidate pairs of this dataset (stage-2 hard negatives), as a sequence classifier with a LoRA
adapter. It is used only on the uncertain pairs of the final decision and is calibrated on
labelled held-out pairs first. Same inputs as the cross-encoder: raw "name | address" texts.

  train : python llm_classifier.py train --pairs-file work/s4_ce_train_pairs_full.parquet --n 300000 --out work/llm_lora
  score : python llm_classifier.py score --out work/llm_lora --pairs work/llm_test_pairs.parquet --dest work/llm_test_lora.parquet
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import Config
from cross_encoder import LengthBucketSampler, record_texts


def fmt(a, b):
    return f"Record A: {a}\nRecord B: {b}\nSame business?"


class Pairs(Dataset):
    def __init__(self, texts, y=None):
        self.t, self.y = texts, y

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], (self.y[i] if self.y is not None else 0.0)


def make_collate(tok, max_len):
    def collate(batch):
        t, y = zip(*batch)
        enc = tok(list(t), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return collate


def load_model(name, adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    else:
        cfg = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return tok, model


@torch.no_grad()
def predict(model, tok, texts, max_len, bs, device):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.empty(len(texts), dtype=np.float32)
    collate = make_collate(tok, max_len)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = collate([(texts[i], 0.0) for i in idx])
        enc.pop("labels")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.float().squeeze(-1)
        out[idx] = torch.sigmoid(logits).cpu().numpy()
    return out


def cmd_train(a):
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)
    t = time.time()
    pairs = pd.read_parquet(a.pairs_file)
    if a.n and len(pairs) > a.n:
        pairs = pairs.sample(n=a.n, random_state=3)
    ta, tp, _, _ = record_texts(cfg, "train")
    texts = [fmt(x, z) for x, z in zip(ta[pairs["q"].values], tp[pairs["p"].values])]
    y = pairs["y"].values.astype(np.float32)
    print(f"[lora] {len(texts):,} pairs ({y.mean():.3f} positive) in {time.time() - t:.0f}s", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(a.model)
    if device.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)
    lengths = np.fromiter((len(x) for x in texts), dtype=np.int32, count=len(texts))
    dl = DataLoader(Pairs(texts, y), batch_sampler=LengthBucketSampler(lengths, a.bs), num_workers=a.workers,
                    collate_fn=make_collate(tok, a.max_len), pin_memory=device.type == "cuda")
    steps = len(dl)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t = 0, time.time()
    model.train()
    for enc in dl:
        labels = enc.pop("labels").to(device)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.float().squeeze(-1)
        loss = lossf(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0 or step == steps:
            el = time.time() - t
            print(f"[lora] step {step}/{steps} loss {loss.item():.4f} {step * a.bs / el:.1f} pairs/s eta {(steps - step) * el / step / 60:.0f} min", flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("[lora] saved", a.out, flush=True)


def cmd_score(a):
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(a.model, adapter=a.out)
    model.to(device)
    df = pd.read_parquet(a.pairs)
    if "a" in df.columns and "b" in df.columns:
        texts = [fmt(x, z) for x, z in zip(df["a"].values, df["b"].values)]
    else:
        ta, tp, _, _ = record_texts(cfg, a.split)
        texts = [fmt(x, z) for x, z in zip(ta[df["q"].values], tp[df["p"].values])]
    t = time.time()
    out = np.empty(len(df), dtype=np.float32)
    step = 50_000
    for s in range(0, len(df), step):
        out[s:s + step] = predict(model, tok, texts[s:s + step], a.max_len, a.bs, device)
        print(f"[lora-score] {min(s + step, len(df)):,}/{len(df):,} {(s + len(out[s:s + step])) / (time.time() - t):.0f} pairs/s", flush=True)
    df["p_lora"] = out
    df.to_parquet(a.dest, index=False)
    if "y" in df.columns:
        from sklearn.metrics import roc_auc_score
        y = df["y"].values
        print(f"[lora-eval] AUC {roc_auc_score(y, out):.4f} acc@.5 {((out >= .5) == (y == 1)).mean():.4f}", flush=True)
        for col in ("prob2", "ce"):
            if col in df.columns:
                print(f"[lora-eval] {col} AUC {roc_auc_score(y, df[col].fillna(0).values):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pairs-file", default="work/s4_ce_train_pairs_full.parquet")
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--out", default="work/llm_lora")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--dest", default="")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_score(a)


if __name__ == "__main__":
    main()
