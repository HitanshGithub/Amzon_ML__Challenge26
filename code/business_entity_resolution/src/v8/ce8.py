"""V8 reranker (ce8) - group-wise cross-encoder trained on the errors of the v7 stack.

Why a new model. In v7 the cross-encoder score dominates the final stacker (gain 137M vs 3.5M
for the next feature), and the remaining recall loss is almost entirely "true pair scored too
low" (1.71% of true pairs, median stage-3 probability 0.255) plus "record given to the wrong
S1 entity" (0.49%). Both are *ranking* failures inside one entity's candidate list, so:

  1. training items are GROUPS (one S1 entity: 1 positive + K negatives), and the loss is
     BCE + a listwise softmax over the group -> the model is optimised for picking the right
     record out of the look-alikes, which is exactly what assign_best needs;
  2. the groups are mined from where v7 is WRONG (positives the old CE scored low, negatives it
     scored high), so capacity goes to the hard band instead of the 95% of pairs already solved;
  3. the text sees the country and both records are given verbatim, and the training set is
     augmented with label-free synthetic pairs built from the *test* records of every country -
     including France, which has no training labels at all. The French synthetic negatives keep
     the city and the generic name words and change only one discriminative token, which is the
     actual French failure mode ("Association du Archives" vs "Association des Arts").

Commands
  mine   build the group training file (real hard groups + synthetic groups)
  train  fine-tune a backbone on it
  score  score a (q, p) pair list of a split -> parquet with column 'ce8'
"""
import argparse
import json
import os
import random
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import Config
from data import load_gt, raw_paths, read_tsv

# --------------------------------------------------------------------------- text
def record_frames(cfg, split):
    p = raw_paths(cfg, split)
    s1 = read_tsv(p["source1"])
    pool = pd.concat([read_tsv(p["source2"]), read_tsv(p["source3"])], ignore_index=True)
    return s1, pool


def texts_of(df):
    return (df["business_name"].str.strip() + " | " + df["business_address"].str.strip()).values


PREFIX = True


def pair_text(country, t):
    return f"{country}: {t}" if PREFIX else t


# --------------------------------------------------------------------------- synthetic noise
FR_LEGAL = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", "EI", "SNC", "SELARL", "SCOP", "SCM", "EIRL"]
FR_LEGAL_VAR = {
    "SARL": ["Sarl", "sarl", "S.A.R.L.", "[SARL]", "SARL.", "S A R L"],
    "SAS": ["Sas", "S.A.S.", "[SAS]", "sas"],
    "SASU": ["Sasu", "S.A.S.U.", "[SASU]"],
    "EURL": ["Eurl", "E.U.R.L.", "[EURL]", "eurl"],
    "SA": ["S.A.", "Sa"],
    "SCI": ["Sci", "S.C.I."],
    "EI": ["E.I.", "Ei"],
    "SNC": ["S.N.C.", "Snc"],
    "SELARL": ["S.E.L.A.R.L.", "Selarl"],
    "SCOP": ["Scop"], "SCM": ["S.C.M."], "EIRL": ["E.I.R.L."],
}
FR_STREET_ABBR = {
    "rue": ["R.", "R", "RUE", "rue"], "avenue": ["AV", "AV.", "AVE", "AVENUE"],
    "boulevard": ["BD", "BD.", "BLVD", "BOULEVARD"], "impasse": ["IMP", "IMP.", "IMPASSE"],
    "chemin": ["CH", "CH.", "CHEM", "CHEMIN"], "allee": ["ALL", "ALL.", "ALLEE"],
    "allée": ["ALL", "ALL.", "ALLEE", "ALLÉE"], "place": ["PL", "PL.", "PLACE"],
    "route": ["RTE", "RT", "ROUTE"], "quai": ["QU", "QUAI"], "cours": ["CRS", "COURS"],
    "square": ["SQ", "SQUARE"], "passage": ["PAS", "PASS", "PASSAGE"],
    "cite": ["CITE", "CITÉ"], "cité": ["CITE", "CITÉ"], "villa": ["VLA", "VILLA"],
    "sentier": ["SENT", "SENTIER"], "esplanade": ["ESP", "ESPL", "ESPLANADE"],
    "faubourg": ["FG", "FBG", "FAUBOURG"], "montee": ["MON", "MONTEE"],
}
# region <-> department: the two sources disagree on which one goes in the last component
FR_REGION_DEPT = {
    "Hauts-de-France": ["Nord", "Pas-de-Calais"], "Nouvelle-Aquitaine": ["Gironde"],
    "Pays de la Loire": ["Loire-Atlantique"], "Ile-de-France": ["Paris", "Seine-Saint-Denis"],
}
FR_DEPT_REGION = {d: r for r, ds in FR_REGION_DEPT.items() for d in ds}
US_ST_ABBR = {"road": "Rd", "street": "St", "avenue": "Ave", "drive": "Dr", "boulevard": "Blvd",
              "lane": "Ln", "court": "Ct", "place": "Pl", "circle": "Cir", "highway": "Hwy",
              "parkway": "Pkwy", "trail": "Trl", "terrace": "Ter", "suite": "Ste", "north": "N",
              "south": "S", "east": "E", "west": "W"}
IN_ST_ABBR = {"road": "Rd", "street": "St", "nagar": "Ngr", "marg": "Mg", "cross": "Crs",
              "main": "Mn", "layout": "Lyt", "phase": "Ph", "sector": "Sec", "block": "Blk",
              "extension": "Extn", "colony": "Cly"}
LEGAL_US = ["Inc", "Inc.", "LLC", "L.L.C.", "Corp", "Corp.", "Corporation", "Co", "Co.", "Ltd", "Limited"]
LEGAL_IN = ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "Ltd", "Limited", "LLP", "Enterprises", "& Co"]
GENERIC_FR = {"association", "amicale", "club", "ecole", "école", "institut", "clinique", "centre",
              "comite", "comité", "societe", "société", "groupe", "maison", "etablissements",
              "établissements", "ets", "cie", "sport", "sportive", "culture", "jeunes", "parents",
              "lycee", "lycée", "primaire", "secondaire", "sante", "santé", "medical", "médical",
              "de", "du", "des", "la", "le", "les", "et", "aux", "france", "saint", "sainte", "st"}
GENERIC_EN = {"the", "and", "of", "for", "inc", "llc", "corp", "co", "ltd", "limited", "company",
              "group", "services", "service", "solutions", "systems", "enterprises", "holdings",
              "partners", "associates", "center", "centre", "clinic", "school", "store", "shop",
              "pvt", "private", "india", "usa", "trust", "society", "works"}


def _strip_acc(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _add_acc(s, rng):
    """S2/S3 sometimes carry *wrong* accents ('Amicale' -> 'Àmicale'): reproduce that."""
    sub = {"a": "à", "A": "À", "e": "ê", "E": "Ê", "i": "î", "o": "ô", "u": "û", "c": "ç"}
    out, done = [], False
    for ch in s:
        if not done and ch in sub and rng.random() < 0.25:
            out.append(sub[ch]); done = True
        else:
            out.append(ch)
    return "".join(out)


def _typo(s, rng):
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    k = rng.random()
    if k < 0.4:
        return s[:i] + s[i + 1:]                       # drop a letter  (PRESIDENT -> PRESIDET)
    if k < 0.7:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]     # transpose
    return s[:i] + s[i] + s[i:]                        # double a letter


def _name_noise(name, country, rng):
    toks = name.split()
    if not toks:
        return name
    legal = FR_LEGAL if country == "France" else (LEGAL_IN if country == "India" else LEGAL_US)
    out = list(toks)
    # legal-form variation / drop / add
    up = [t.upper().strip(".[]()") for t in out]
    hit = [i for i, t in enumerate(up) if t in FR_LEGAL_VAR or t in {x.upper() for x in legal}]
    r = rng.random()
    if hit and r < 0.35:
        i = hit[-1]
        key = up[i]
        if key in FR_LEGAL_VAR and rng.random() < 0.7:
            out[i] = rng.choice(FR_LEGAL_VAR[key])
        else:
            out.pop(i)
    elif not hit and r < 0.55:
        out.append(rng.choice(legal))
    if country == "France":
        if rng.random() < 0.18:                        # the '(France)' / 'France' token
            out.insert(max(1, len(out) - 1), rng.choice(["(France)", "France"]))
        if rng.random() < 0.20:
            out = [rng.choice(["Ets", "Établissements", "SCI", "Sté"])] + out
    if rng.random() < 0.25:
        out = [t.replace("&", "and") if "&" in t else t for t in out]
    s = " ".join(out)
    if country == "France":
        if rng.random() < 0.30:
            s = _strip_acc(s)
        elif rng.random() < 0.20:
            s = _add_acc(s, rng)
    if rng.random() < 0.22:
        s = s.upper()
    elif rng.random() < 0.12:
        s = s.lower()
    if rng.random() < 0.18:
        w = s.split()
        if len(w) > 2:
            i = rng.randrange(len(w))
            w[i] = _typo(w[i], rng)
            s = " ".join(w)
    if rng.random() < 0.12:                            # word-order transposition
        w = s.split()
        if len(w) > 2:
            i = rng.randrange(len(w) - 1)
            w[i], w[i + 1] = w[i + 1], w[i]
            s = " ".join(w)
    return s


def _addr_noise(addr, country, rng):
    if not addr.strip():
        return addr
    if rng.random() < 0.05:
        return ""                                      # the empty-address records (2.7% of the pool)
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    if country == "France":
        for i, c in enumerate(comps):
            if c in FR_DEPT_REGION and rng.random() < 0.5:
                comps[i] = FR_DEPT_REGION[c]
            elif c in FR_REGION_DEPT and rng.random() < 0.5:
                comps[i] = rng.choice(FR_REGION_DEPT[c])
        if rng.random() < 0.25:
            comps = [c for c in comps if c not in FR_DEPT_REGION and c not in FR_REGION_DEPT] or comps
    if rng.random() < 0.30 and len(comps) > 1:         # component reordering
        rng.shuffle(comps)
    if rng.random() < 0.18 and len(comps) > 2:         # missing component (no PIN / no state)
        comps.pop(rng.randrange(len(comps)))
    s = ", ".join(comps)
    tbl = FR_STREET_ABBR if country == "France" else None
    w = s.split()
    for i, t in enumerate(w):
        low = t.lower().strip(".,")
        if tbl and low in tbl and rng.random() < 0.6:
            w[i] = rng.choice(tbl[low])
        elif not tbl:
            m = US_ST_ABBR if country == "US" else IN_ST_ABBR
            if low in m and rng.random() < 0.5:
                w[i] = m[low]
    s = " ".join(w)
    if country == "France":
        if rng.random() < 0.30:
            s = _strip_acc(s)
        if rng.random() < 0.20:
            s = re.sub(r"\b(\d+)\s+bis\b", r"\1 B", s, flags=re.I)
        if rng.random() < 0.10:
            s = "NO. " + s
    if rng.random() < 0.30:
        s = s.upper()
    if rng.random() < 0.12:
        w = s.split()
        if len(w) > 2:
            i = rng.randrange(len(w))
            w[i] = _typo(w[i], rng)
            s = " ".join(w)
    if country == "India" and rng.random() < 0.12:
        s = s + ", " + rng.choice(["Near SBI ATM", "Opp. Bus Stand", "Behind Temple", "Near Metro Station"])
    return s


def _hard_neg_name(name, country, other_names, rng):
    """Change ONE discriminative token, keep the generic scaffolding -> the real ambiguity."""
    generic = GENERIC_FR if country == "France" else GENERIC_EN
    toks = name.split()
    disc = [i for i, t in enumerate(toks)
            if t.lower().strip(".,()[]&") not in generic and len(t) > 2 and not t.isupper()]
    if not disc:
        disc = [i for i in range(len(toks)) if len(toks[i]) > 2]
    if not disc:
        return None
    i = rng.choice(disc)
    for _ in range(8):
        cand = rng.choice(other_names).split()
        pool_t = [t for t in cand if t.lower().strip(".,()[]&") not in generic and len(t) > 2]
        if pool_t:
            new = list(toks)
            new[i] = rng.choice(pool_t)
            if " ".join(new) != name:
                return " ".join(new)
    return None


def _bump_number(addr, rng):
    m = list(re.finditer(r"\b(\d{1,4})\b", addr))
    if not m:
        return None
    g = rng.choice(m)
    n = int(g.group(1))
    new = n + rng.choice([-3, -2, -1, 1, 2, 3, 10])
    if new <= 0:
        new = n + 7
    return addr[:g.start(1)] + str(new) + addr[g.end(1):]


def synth_groups(cfg, n_per_country, k_neg, seed=0):
    """Label-free synthetic groups from the TEST records of every country (France included)."""
    rng = random.Random(seed)
    s1, pool = record_frames(cfg, "test")
    rows = []
    gid = 0
    for country, share in (("France", 0.5), ("India", 0.25), ("US", 0.25)):
        sub = s1[s1["country"] == country]
        if not len(sub):
            continue
        n = int(n_per_country * 4 * share)
        take = sub.sample(min(n, len(sub)), random_state=seed)
        names = sub["business_name"].str.strip().values
        addrs = sub["business_address"].str.strip().values
        nm_list = names.tolist()
        for nm, ad in zip(take["business_name"].str.strip().values, take["business_address"].str.strip().values):
            a = pair_text(country, f"{nm} | {ad}")
            pos = f"{_name_noise(nm, country, rng)} | {_addr_noise(ad, country, rng)}"
            rows.append((gid, a, pair_text(country, pos), 1.0))
            got = 0
            for _ in range(k_neg * 3):
                if got >= k_neg:
                    break
                r = rng.random()
                if r < 0.45:                                   # one discriminative token changed
                    nn = _hard_neg_name(nm, country, nm_list, rng)
                    if nn is None:
                        continue
                    b = f"{_name_noise(nn, country, rng)} | {_addr_noise(ad, country, rng)}"
                elif r < 0.75:                                 # same name, different street number
                    na = _bump_number(ad, rng)
                    if na is None:
                        continue
                    b = f"{_name_noise(nm, country, rng)} | {_addr_noise(na, country, rng)}"
                else:                                          # same name, another address of the country
                    j = rng.randrange(len(addrs))
                    if addrs[j] == ad:
                        continue
                    b = f"{_name_noise(nm, country, rng)} | {_addr_noise(addrs[j], country, rng)}"
                rows.append((gid, a, pair_text(country, b), 0.0))
                got += 1
            gid += 1
    df = pd.DataFrame(rows, columns=["gid", "a", "b", "y"])
    print(f"[synth] {gid:,} groups, {len(df):,} rows ({df.y.mean():.3f} positive)", flush=True)
    return df


# --------------------------------------------------------------------------- mining
def cmd_mine(cfg, a):
    t = time.time()
    s1, pool = record_frames(cfg, "train")
    ctry = s1["country"].values
    ts1 = texts_of(s1)
    tpool = texts_of(pool)
    n_pool = len(pool)

    u = np.random.default_rng(cfg.seed).random(len(s1))
    train_mask = u >= cfg.val_s1_frac                           # never mine validation entities
    gt = load_gt(cfg)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(n_pool), index=pool["entity_id"].values)
    gt = gt[gt["entity_id"].isin(p_pos.index)]
    gq, gp = s1_pos[gt["source1_entity_id"]].values, p_pos[gt["entity_id"]].values
    tk = gq.astype(np.int64) * n_pool + gp
    print(f"[mine] {len(gq):,} true pairs; {time.time()-t:.0f}s", flush=True)

    ce = pd.read_parquet(a.ce_file)                             # v7 cross-encoder scores (q, p, ce)
    ce = ce[train_mask[ce["q"].values]]
    ce["y"] = np.isin(ce["q"].values.astype(np.int64) * n_pool + ce["p"].values, tk)
    rng = np.random.default_rng(a.seed)
    # hard = the v7 CE got it wrong or was unsure; easy = keep some for calibration
    hard_pos = ce[(ce["y"]) & (ce["ce"] < a.hi)]
    hard_neg = ce[(~ce["y"]) & (ce["ce"] > a.lo)]
    easy_pos = ce[(ce["y"]) & (ce["ce"] >= a.hi)].sample(frac=a.easy_frac, random_state=a.seed)
    easy_neg = ce[(~ce["y"]) & (ce["ce"] <= a.lo)].sample(frac=a.easy_frac * 0.5, random_state=a.seed)
    seed_rows = pd.concat([hard_pos, hard_neg, easy_pos, easy_neg], ignore_index=True)
    print(f"[mine] hard_pos {len(hard_pos):,} hard_neg {len(hard_neg):,} easy_pos {len(easy_pos):,} "
          f"easy_neg {len(easy_neg):,}; {time.time()-t:.0f}s", flush=True)
    # A group must contain the whole competition, not just the one hard row: pick the ENTITIES that
    # own a hard row, then take all of their cross-encoder candidates. Listwise loss needs rivals.
    seed_q = seed_rows["q"].unique()
    if a.max_q and len(seed_q) > a.max_q:
        seed_q = rng.choice(seed_q, size=a.max_q, replace=False)
    if a.real_q:                                   # + ordinary entities with their whole candidate list
        allq = ce["q"].unique()
        extra = rng.choice(allq, size=min(a.real_q, len(allq)), replace=False)
        seed_q = np.unique(np.concatenate([seed_q, extra]))
    sel = ce[np.isin(ce["q"].values, seed_q)].copy()
    sel["o"] = -(sel["y"].to_numpy().astype(np.int8) * 2 + (np.abs(sel["ce"].to_numpy() - 0.5) < 0.45))
    sel = sel.sort_values(["q", "o"])                       # positives first, then the uncertain rivals
    sel = sel[sel.groupby("q").cumcount() < a.max_per_q]
    sel = sel.sort_values(["q", "y"], ascending=[True, False]).reset_index(drop=True)
    gid = sel.groupby("q").ngroup().to_numpy()
    real = pd.DataFrame({"gid": gid, "a": [pair_text(ctry[q], ts1[q]) for q in sel["q"].values],
                         "b": [pair_text(ctry[q], tpool[p]) for q, p in zip(sel["q"].values, sel["p"].values)],
                         "y": sel["y"].values.astype(np.float32)})
    print(f"[mine] real groups {real.gid.nunique():,} rows {len(real):,} ({real.y.mean():.3f} pos); "
          f"{time.time()-t:.0f}s", flush=True)

    syn = synth_groups(cfg, a.syn_per_country, a.syn_neg, seed=a.seed)
    syn["gid"] = syn["gid"] + real["gid"].max() + 1
    out = pd.concat([real, syn], ignore_index=True)
    out.to_parquet(cfg.path(a.out), index=False)
    print(f"[mine] wrote {cfg.path(a.out)}: {len(out):,} rows, {out.gid.nunique():,} groups; "
          f"{time.time()-t:.0f}s", flush=True)
    print("MINE_DONE", flush=True)


# --------------------------------------------------------------------------- model
class GroupDS(Dataset):
    def __init__(self, df):
        self.a = df["a"].tolist(); self.b = df["b"].tolist(); self.y = df["y"].to_numpy(np.float32)
        self.gid = df["gid"].to_numpy()
        bounds = np.flatnonzero(np.diff(self.gid)) + 1
        self.groups = np.split(np.arange(len(self.gid)), bounds)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        g = self.groups[i]
        return [(self.a[j], self.b[j], self.y[j]) for j in g]


def make_collate(tok, max_len):
    def collate(batch):
        flat, gsz = [], []
        for grp in batch:
            flat += grp; gsz.append(len(grp))
        a, b, y = zip(*flat)
        enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        enc["gsz"] = torch.tensor(gsz, dtype=torch.long)
        return enc
    return collate


def load_backbone(name, device, train=True):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(num_labels=1, dtype=torch.bfloat16 if train else torch.bfloat16)
    m = AutoModelForSequenceClassification.from_pretrained(name, **kw)
    if m.config.pad_token_id is None:
        m.config.pad_token_id = tok.pad_token_id
    return tok, m.to(device)


def cmd_train(cfg, a):
    t = time.time()
    device = "cuda"
    df = pd.read_parquet(cfg.path(a.pairs))
    # hold out 2% of groups to report honest numbers
    gids = df["gid"].unique()
    rng = np.random.default_rng(0)
    hold = set(rng.choice(gids, size=max(1, int(0.02 * len(gids))), replace=False).tolist())
    va = df[df["gid"].isin(hold)].reset_index(drop=True)
    tr = df[~df["gid"].isin(hold)].reset_index(drop=True)
    tok, model = load_backbone(a.init or a.model, device)
    model.gradient_checkpointing_disable()
    ds = GroupDS(tr)
    dl = DataLoader(ds, batch_size=a.groups_per_batch, shuffle=True, collate_fn=make_collate(tok, a.max_len),
                    num_workers=4, drop_last=True, pin_memory=True)
    steps = len(dl) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    print(f"[train] {a.model}: {len(tr):,} rows / {len(ds):,} groups, {steps:,} steps; {time.time()-t:.0f}s", flush=True)
    model.train()
    seen = 0
    for ep in range(a.epochs):
        for i, enc in enumerate(dl):
            gsz = enc.pop("gsz")
            y = enc.pop("labels").to(device)
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            logit = model(**enc).logits.squeeze(-1).float()
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if a.listwise > 0:
                # InfoNCE per group: every positive must beat that entity's negatives. Groups can
                # hold several positives (one business has 2-4 copies), so each gets its own term.
                off, ls, n = 0, 0.0, 0
                for g in gsz.tolist():
                    lg, yg = logit[off:off + g], y[off:off + g]
                    off += g
                    pos = (yg > 0.5).nonzero(as_tuple=True)[0]
                    neg = (yg <= 0.5).nonzero(as_tuple=True)[0]
                    if len(pos) == 0 or len(neg) == 0:
                        continue
                    ln = lg[neg]
                    for ip in pos:
                        ls = ls - (lg[ip] - torch.logsumexp(torch.cat([lg[ip:ip + 1], ln]), 0))
                        n += 1
                if n:
                    loss = loss + a.listwise * ls / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            seen += len(y)
            if i % 500 == 0:
                print(f"[train] ep{ep} step {i}/{len(dl)} loss {loss.item():.4f} rows {seen:,} "
                      f"{time.time()-t:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    # honest validation on the held-out groups
    sc = predict_texts(model, tok, va["a"].tolist(), va["b"].tolist(), a.max_len, a.bs, device)
    y = va["y"].to_numpy()
    eps = 1e-6
    ll = -(y * np.log(sc + eps) + (1 - y) * np.log(1 - sc + eps)).mean()
    va = va.assign(sc=sc)
    ok = va[va.groupby("gid")["y"].transform("max") > 0.5]
    top1 = ok.loc[ok.groupby("gid")["sc"].idxmax(), "y"].mean() if len(ok) else float("nan")
    rep = {"rows": int(len(va)), "logloss": float(ll), "group_top1_acc": float(top1),
           "mean_pos": float(sc[y > 0.5].mean()), "mean_neg": float(sc[y < 0.5].mean()),
           "pos_below_0.5": float((sc[y > 0.5] < 0.5).mean()), "neg_above_0.5": float((sc[y < 0.5] > 0.5).mean())}
    json.dump(rep, open(os.path.join(a.out, "holdout.json"), "w"), indent=1)
    print("[train] holdout", json.dumps(rep), flush=True)
    print("TRAIN_DONE", flush=True)


class _ScoreDS(Dataset):
    """Length-sorted batches; tokenisation happens in DataLoader workers, not the GPU loop."""
    def __init__(self, a_txt, b_txt, order, bs):
        self.a, self.b, self.batches = a_txt, b_txt, [order[s:s + bs] for s in range(0, len(order), bs)]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        return idx, [self.a[j] for j in idx], [self.b[j] for j in idx]


@torch.no_grad()
def predict_texts(model, tok, a_txt, b_txt, max_len, bs, device, workers=6):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a_txt, b_txt)])
    out = np.empty(len(a_txt), dtype=np.float32)

    def collate(items):
        idx, a, b = items[0]
        enc = tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return idx, enc
    dl = DataLoader(_ScoreDS(a_txt, b_txt, order, bs), batch_size=1, shuffle=False, num_workers=workers,
                    collate_fn=collate, prefetch_factor=4 if workers else None, pin_memory=True)
    for idx, enc in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**enc).logits.squeeze(-1).float()
        out[idx] = torch.sigmoid(lg).cpu().numpy()
    model.train()
    return out


def cmd_score(cfg, a):
    t = time.time()
    device = "cuda"
    tok, model = load_backbone(a.out, device, train=False)
    model.eval()
    s1, pool = record_frames(cfg, a.split)
    ctry = s1["country"].values
    ts1, tpool = texts_of(s1), texts_of(pool)
    pairs = pd.read_parquet(a.pairs_file)
    if a.limit:
        pairs = pairs.iloc[:a.limit]
    q, p = pairs["q"].to_numpy(), pairs["p"].to_numpy()
    print(f"[score] {len(pairs):,} pairs on {a.split}; {time.time()-t:.0f}s", flush=True)
    res = np.empty(len(pairs), dtype=np.float32)
    step = 2_000_000
    for s in range(0, len(pairs), step):
        e = min(s + step, len(pairs))
        at = [pair_text(ctry[i], ts1[i]) for i in q[s:e]]
        bt = [pair_text(ctry[i], tpool[j]) for i, j in zip(q[s:e], p[s:e])]
        res[s:e] = predict_texts(model, tok, at, bt, a.max_len, a.bs, device)
        print(f"[score] {e:,}/{len(pairs):,} ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame({"q": q, "p": p, "ce8": res}).to_parquet(cfg.path(a.dest), index=False)
    print(f"[score] wrote {cfg.path(a.dest)}; {time.time()-t:.0f}s", flush=True)
    print("SCORE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "train", "score", "synth"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", default="work/ce8_e5")
    ap.add_argument("--pairs", default="ce8_groups.parquet")
    ap.add_argument("--pairs-file", default="")
    ap.add_argument("--dest", default="ce8_scores.parquet")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ce-file", default="work/v7_train_ce_fr.parquet")
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--easy-frac", type=float, default=0.10)
    ap.add_argument("--max-per-q", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-q", type=int, default=250_000)
    ap.add_argument("--real-q", type=int, default=0, help="extra random train entities with full groups")
    ap.add_argument("--init", default="", help="continue from this local cross-encoder directory")
    ap.add_argument("--no-prefix", action="store_true", help="plain 'name | address' text (v7 CE format)")
    ap.add_argument("--syn-per-country", type=int, default=40_000)
    ap.add_argument("--syn-neg", type=int, default=5)
    ap.add_argument("--groups-per-batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=112)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--listwise", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    global PREFIX
    if a.no_prefix:
        PREFIX = False
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)
    {"mine": cmd_mine, "train": cmd_train, "score": cmd_score,
     "synth": lambda c, x: synth_groups(c, x.syn_per_country, x.syn_neg).to_parquet(c.path("syn_only.parquet"))}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
