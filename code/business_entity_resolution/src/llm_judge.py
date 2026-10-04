"""LLM judge for uncertain candidate pairs (Qwen2.5-7B-Instruct, Apache-2.0, 7.6B parameters).

The pipeline's stage-3 model is confident on ~95% of pairs; the rest are evidence-poor or
ambiguous (co-located entities, replacement names, one-digit address noise). This judge reads the
two raw records and returns P(same business) from the yes/no next-token log-probabilities. It is
validated on labelled US/India uncertain pairs before being applied to the test set; no external
data or lookups are involved - the model only sees the two records.

  judge : python llm_judge.py judge --pairs llm_val_pairs.parquet --out llm_val_scored.parquet
  eval  : python llm_judge.py eval  --scored llm_val_scored.parquet
"""
import argparse
import math
import time

import numpy as np
import pandas as pd

SYSTEM = (
    "You decide whether two business records from different noisy data sources describe the same "
    "real-world business. Names may be abbreviated, misspelled, transliterated from another script, "
    "reordered, missing the legal form, written as a website, given as a trade/DBA name, or replaced by "
    "a meaningless made-up word; addresses may be partial, abbreviated, missing, reordered, or contain a "
    "small house-number error. Different businesses can share the same address, so a matching address "
    "alone is not proof; a completely different real business name at the same address is usually a "
    "different business, while a made-up word or a website name with the same address is usually the "
    "same business. Answer with exactly one word: yes or no."
)


def build_prompts(tok, A, B):
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Record A: {a}\nRecord B: {b}\nSame business?"}] for a, b in zip(A, B)]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]


def cmd_judge(a):
    from vllm import LLM, SamplingParams
    df = pd.read_parquet(a.pairs)
    if a.limit:
        df = df.head(a.limit)
    llm = LLM(model=a.model, dtype="half", max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes", "YES")}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No", "NO")}
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    out = np.full(len(df), np.nan, dtype=np.float32)
    t = time.time()
    for s in range(0, len(df), a.batch):
        chunk = df.iloc[s:s + a.batch]
        prompts = build_prompts(tok, chunk["a"].tolist(), chunk["b"].tolist())
        res = llm.generate(prompts, sp, use_tqdm=False)
        for i, r in enumerate(res):
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            py = max((v.logprob for k, v in lp.items() if k in yes_ids), default=-30.0)
            pn = max((v.logprob for k, v in lp.items() if k in no_ids), default=-30.0)
            out[s + i] = 1.0 / (1.0 + math.exp(pn - py))
        print(f"[llm] {min(s + a.batch, len(df)):,}/{len(df):,} pairs, {(s + len(chunk)) / (time.time() - t):.1f} pairs/s", flush=True)
    df["p_llm"] = out
    df.to_parquet(a.out, index=False)


def cmd_eval(a):
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(a.scored)
    df = df[df["p_llm"].notna()]
    y = df["y"].values
    print(f"pairs {len(df):,}, positive share {y.mean():.3f}")
    cols = {"stage2": df["prob2"].fillna(0).values, "cross-encoder": df["ce"].fillna(0).values, "llm": df["p_llm"].values}
    cols["mean(ce,llm)"] = (cols["cross-encoder"] + cols["llm"]) / 2
    cols["mean(all)"] = (cols["stage2"] + cols["cross-encoder"] + cols["llm"]) / 3
    for name, v in cols.items():
        acc = ((v >= 0.5) == (y == 1)).mean()
        prec = y[v >= 0.5].mean() if (v >= 0.5).any() else float("nan")
        rec = (v[y == 1] >= 0.5).mean()
        print(f"  {name:14s} AUC {roc_auc_score(y, v):.4f}  acc@.5 {acc:.4f}  precision@.5 {prec:.4f}  recall@.5 {rec:.4f}")
    if "country" in df.columns:
        for c, g in df.groupby("country"):
            print(f"  [{c}] n={len(g):,}: AUC stage2 {roc_auc_score(g.y, g.prob2.fillna(0)):.4f} | ce {roc_auc_score(g.y, g.ce.fillna(0)):.4f} | llm {roc_auc_score(g.y, g.p_llm):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "eval"])
    ap.add_argument("--pairs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--scored", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cmd_judge(a) if a.cmd == "judge" else cmd_eval(a)


if __name__ == "__main__":
    main()
