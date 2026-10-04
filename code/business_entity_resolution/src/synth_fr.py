"""Synthetic labelled pairs for a country that has no training labels (France).

Only provided data is used: the unlabelled French Source-1 records of the test set are
corrupted with the noise operations measured on real French S2/S3 copies (and on the
US/India training pairs). Positives = (S1 record, noisy copy of the SAME record);
negatives = (S1 record, noisy copy of a LOOK-ALIKE French S1 record: same address,
same/similar name, or same street). Output: parquet with columns a, b, y (raw texts), fed to
cross_encoder.py train --text-pairs.
"""
import argparse
import csv
import random
import re
import unicodedata

import numpy as np
import pandas as pd

LEGAL_VARIANTS = {
    "SARL": ["SARL", "Sarl", "S.A.R.L.", "(SARL)", "[SARL]", "Sàrl", "sarl"],
    "SAS": ["SAS", "Sas", "S.A.S.", "(SAS)", "S.A.S", "(S.A.S)", "[SAS]"],
    "SASU": ["SASU", "Sasu", "S.A.S.U.", "(SASU)"],
    "EURL": ["EURL", "Eurl", "E.U.R.L.", "(EURL)"],
    "SA": ["SA", "S.A.", "(SA)"],
    "SCI": ["SCI", "S.C.I.", "Sci", "(SCI)"],
    "EI": ["EI", "E.I.", "[EI]", "(EI)"],
}
# measured on 853,886 high-confidence French matches: words the noise process adds (weights = counts)
ADD_WEIGHTED = [("Et Fils", 5200), ("Fils", 5100), ("Cie", 7600), ("Services", 7600), ("France", 5500),
                ("& Associés", 4700), ("Groupe", 2900), ("Développement", 2600), ("Club", 2100), ("Comite", 1950),
                ("Ecole", 1770), ("Compagnie", 1560), ("Amicale", 1500), ("Amis", 1270), ("Centre", 1120),
                ("Trading", 1050), ("Societe", 830), ("Parents", 740), ("Fetes", 715), ("Labs", 700), ("Sys", 680),
                ("One", 660), ("Sante", 550), ("Primaire", 550), ("College", 540), ("Federation", 520),
                ("Participations", 400), ("Holding", 400), ("Distribution", 350)]
ADD_WORDS = [w for w, _ in ADD_WEIGHTED]
ADD_P = [c for _, c in ADD_WEIGHTED]
# category words the noise process drops or replaces (measured), and French abbreviation swaps
CATEGORY = {"club", "france", "ecole", "école", "amicale", "comite", "comité", "sportive", "centre", "amis", "maison",
            "union", "freres", "frères", "primaire", "college", "collège", "cie", "parents", "sante", "santé", "fils",
            "federation", "fédération", "pharmacie", "societe", "société", "compagnie", "maternelle", "fetes", "fêtes",
            "lycee", "lycée", "anciens", "loisirs", "de", "du", "des"}
REPLACE_WITH = [("Services", 5), ("Cie", 5), ("Groupe", 3), ("Fils", 3), ("France", 2), ("Associés", 3), ("Comite", 1), ("Ecole", 1)]
ABBR = {"compagnie": "Cie", "cie": "Compagnie", "frères": "Frs", "freres": "Frs", "saint": "St", "sainte": "Ste",
        "club": "CB", "établissements": "Ets", "etablissements": "Ets", "société": "Sté", "association": "Assoc"}
SYLL = ["nyla", "zeph", "umbra", "vio", "halo", "kelo", "calo", "xylo", "vera", "arc", "flux", "pyra",
        "quo", "faye", "yuma", "belo", "lum", "dova", "tavo", "zeta", "orbi", "ciran", "eto", "brix",
        "novi", "delta", "mira", "syn", "gild", "vantage", "aria", "lyra", "onyx", "evo", "fly"]
STREET = {"Rue": ["R.", "R", "RUE", "rue"], "Avenue": ["Av", "Av.", "AV", "Ave", "AVENUE"],
          "Boulevard": ["Bd", "Bd.", "BD", "Blvd", "BOULEVARD"], "Allée": ["All.", "Allee", "All", "ALLÉE"],
          "Impasse": ["Imp.", "Imp", "IMP"], "Place": ["Pl.", "Pl", "PL"], "Chemin": ["Ch.", "Chem.", "CH"],
          "Route": ["Rte", "Rte.", "RTE"], "Cours": ["Crs", "CRS"], "Square": ["Sq.", "SQ"], "Quai": ["Qu.", "QU"]}
REGION_DEPT = {"bordeaux": "Gironde", "pessac": "Gironde", "mérignac": "Gironde", "merignac": "Gironde",
               "lège-cap-ferret": "Gironde", "la teste-de-buch": "Gironde", "lille": "Nord", "tourcoing": "Nord",
               "roubaix": "Nord", "dunkerque": "Nord", "calais": "Pas-de-Calais", "nantes": "Loire-Atlantique",
               "saint-nazaire": "Loire-Atlantique", "pornic": "Loire-Atlantique", "saint-herblain": "Loire-Atlantique",
               "la baule-escoublac": "Loire-Atlantique"}
REGIONS = {"hauts-de-france", "nouvelle-aquitaine", "pays de la loire"}


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def typo(w, r):
    if len(w) < 4:
        return w
    i = r.randrange(1, len(w) - 1)
    op = r.random()
    if op < 0.35:
        return w[:i] + r.choice("abcdefghilmnoprstu") + w[i + 1:]
    if op < 0.6:
        return w[:i] + w[i + 1:]
    if op < 0.8:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + r.choice("aeinrst") + w[i:]


def accent_noise(w, r):
    m = {"a": "à", "e": "é", "o": "ô", "c": "ç", "u": "ù", "i": "ï", "C": "Ç", "E": "É", "A": "À"}
    idx = [i for i, c in enumerate(w) if c in m]
    if not idx:
        return w
    i = r.choice(idx)
    return w[:i] + m[w[i]] + w[i + 1:]


def synth_name(r):
    n = r.choice([2, 2, 3])
    s = "".join(r.choice(SYLL) for _ in range(n))
    return s.capitalize() if r.random() < 0.7 else s.upper()


def noisy_name(name, r, src):
    """Returns (noisy name, heavy) - heavy = the name no longer identifies the business on its own."""
    toks = name.split()
    legal = [t for t in toks if t.upper() in LEGAL_VARIANTS]
    core = [t for t in toks if t.upper() not in LEGAL_VARIANTS]
    u = r.random()
    if u < 0.02:
        return synth_name(r), True
    if u < 0.035:
        return f"{synth_name(r)} {r.choice(['DBA', 'dba', 'DBA:', 'f/k/a', 't/a', 'a/k/a', 'formerly:'])} {name}", False
    if u < 0.088:
        return strip_acc("".join(core + legal)).lower().replace("'", "").replace("&", "") + r.choice([".com", ".fr", ".com"]), True
    if u < 0.10 and len(core) >= 2:
        return "".join(t[0] for t in core if t[0].isalpha()).upper(), True
    heavy = False
    cat = [i for i, t in enumerate(core) if t.lower() in CATEGORY]
    if r.random() < 0.10 and cat:
        i = r.choice(cat); core[i] = r.choices([w for w, _ in REPLACE_WITH], [c for _, c in REPLACE_WITH])[0]
    elif r.random() < 0.03 and len(core) >= 2:
        i = r.randrange(len(core)); core[i] = r.choices(ADD_WORDS, ADD_P)[0]; heavy = True
    for i, t in enumerate(core):
        if t.lower() in ABBR and r.random() < 0.12:
            core[i] = ABBR[t.lower()]
    if r.random() < 0.10 and len(core) >= 2:
        i, j = r.sample(range(len(core)), 2); core[i], core[j] = core[j], core[i]
    if r.random() < 0.20:
        w = r.choices(ADD_WORDS, ADD_P)[0]
        core.insert(len(core) if r.random() < 0.7 else r.randrange(len(core) + 1), w)
    cat = [i for i, t in enumerate(core) if t.lower() in CATEGORY]
    if r.random() < 0.14 and len(core) >= 2 and cat:
        core.pop(r.choice(cat))
    if r.random() < 0.14 and core:
        i = r.randrange(len(core)); core[i] = typo(core[i], r)
    if r.random() < 0.10 and core:
        i = r.randrange(len(core)); core[i] = accent_noise(core[i], r)
    if r.random() < 0.04 and core:
        core.insert(0, "(France)" if r.random() < 0.5 else "France")
    new_legal = []
    for l in legal:
        v = r.random()
        if v < 0.35:
            continue
        new_legal.append(r.choice(LEGAL_VARIANTS[l.upper()]))
    if r.random() < 0.03:
        new_legal.append(r.choice(list(LEGAL_VARIANTS)))
    out = (new_legal + core) if (new_legal and r.random() < 0.15) else (core + new_legal)
    s = " ".join(out)
    if r.random() < 0.07:
        s = s.replace(" ", "  ", 1)
    if r.random() < 0.01:
        s = r.choice(["-- ", "... ", "@", "#", "<< "]) + s
    if r.random() < 0.02 and core:
        s = s.replace(core[-1], f"[{core[-1]}]", 1)
    case = r.random()
    if src == 2 and case < 0.20:
        s = s.upper()
    elif case < 0.27 if src == 2 else case < 0.07:
        s = s.lower()
    return s, heavy


def noisy_addr(addr, r, src, allow_empty=True):
    if allow_empty and r.random() < 0.03:
        return ""
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    region = [c for c in comps if c.lower() in REGIONS]
    rest = [c for c in comps if c.lower() not in REGIONS]
    city = None
    for c in rest:
        if c.lower() in REGION_DEPT:
            city = c
    out = []
    for c in rest:
        if c == city:
            continue
        m = re.match(r"^(\d+)\s*(bis|ter|b|BIS|Bis)?\s*(.*)$", c)
        if m and m.group(1):
            num, suf, street = m.group(1), m.group(2) or "", m.group(3)
            v = r.random()
            if v < 0.07:
                num = ""
            elif v < 0.12:
                num = str(max(1, int(num) + r.choice([-4, -2, -1, 1, 2, 3, 11, 20])))
            elif v < 0.17:
                num = num.zfill(r.choice([3, 5]))
            if suf and r.random() < 0.4:
                suf = r.choice(["B", "b", "Bis", "BIS", "bis"]) if suf.lower().startswith("b") else suf
            pre = ""
            p = r.random()
            if num and p < 0.07:
                pre = r.choice(["N° ", "Nº ", "No. ", "No ", "N°"])
            elif num and p < 0.10:
                num = f"({num})"
            elif num and p < 0.13:
                pre = "#"
            words = street.split()
            if words and words[0] in STREET and r.random() < 0.55:
                words[0] = r.choice(STREET[words[0]])
            if len(words) > 1 and r.random() < 0.12:
                i = r.randrange(1, len(words)); words[i] = typo(words[i], r)
            c = " ".join(x for x in [pre + num + (suf if suf and len(suf) == 1 and num else (" " + suf if suf else "")), " ".join(words)] if x).strip()
        out.append(c)
    if city:
        cv = city
        if r.random() < 0.15:
            cv = cv.replace("Saint-", r.choice(["St-", "St.-", "Saint "]))
        if r.random() < 0.12:
            cv = cv.replace("-", " ")
        if r.random() < 0.2:
            cv = cv.capitalize() if "-" in cv else cv
        out.append(cv)
    reg = r.random()
    if region:
        if reg < 0.30 and city:
            out.append(REGION_DEPT.get(city.lower(), region[0]))
        elif reg < 0.70:
            out.append(region[0])
    if r.random() < 0.18 and len(out) > 1:
        r.shuffle(out)
    s = ", ".join(out)
    if r.random() < 0.30:
        s = strip_acc(s)
    if src == 2 and r.random() < 0.62:
        s = s.upper()
    elif r.random() < 0.15:
        s = s.title()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="dataset/test/test_source1.tsv")
    ap.add_argument("--country", default="France")
    ap.add_argument("--n-s1", type=int, default=150_000)
    ap.add_argument("--out", default="work/synth_fr_pairs.parquet")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    r = random.Random(a.seed)
    d = pd.read_csv(a.s1, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
    d = d[d["country"] == a.country].reset_index(drop=True)
    fmt = lambda n, ad: f"{n} | {ad}"

    def copy_of(j, src):
        nm, heavy = noisy_name(d.business_name[j], r, src)
        return fmt(nm, noisy_addr(d.business_address[j], r, src, allow_empty=not heavy))
    s1txt = [fmt(n, ad) for n, ad in zip(d.business_name, d.business_address)]
    # look-alike groups: same exact address, same first name token + city, same street
    key_addr = d.business_address.str.lower().values
    first = d.business_name.str.split().str[0].str.lower().values
    street = d.business_address.str.replace(r"^\s*\d+\s*(bis|ter|b)?\s*", "", regex=True).str.split(",").str[0].str.lower().values
    by_addr, by_first, by_street = {}, {}, {}
    for i in range(len(d)):
        by_addr.setdefault(key_addr[i], []).append(i)
        by_first.setdefault(first[i], []).append(i)
        by_street.setdefault(street[i], []).append(i)
    idx = list(range(len(d)))
    r.shuffle(idx)
    rows = []
    for i in idx[:a.n_s1]:
        for _ in range(r.choice([1, 2, 2])):
            src = r.choice([2, 3])
            rows.append((s1txt[i], copy_of(i, src), 1.0))
        for grp in (by_addr[key_addr[i]], by_first[first[i]], by_street[street[i]]):
            others = [j for j in grp if j != i]
            if others:
                j = r.choice(others)
                src = r.choice([2, 3])
                rows.append((s1txt[i], copy_of(j, src), 0.0))
    out = pd.DataFrame(rows, columns=["a", "b", "y"]).sample(frac=1.0, random_state=a.seed)
    out.to_parquet(a.out, index=False)
    print(f"[synth] {len(out):,} pairs ({out.y.mean():.3f} positive) from {min(a.n_s1, len(d)):,} {a.country} S1 records", flush=True)
    for _, x in out.head(12).iterrows():
        print(f"  y={int(x.y)}  {x.a}  <->  {x.b}")


if __name__ == "__main__":
    main()
