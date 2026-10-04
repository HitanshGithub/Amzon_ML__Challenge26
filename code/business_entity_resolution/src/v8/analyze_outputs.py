"""Compare every full-size output_*/matching_results.tsv in the folder."""
import csv, glob, os, collections, itertools, time
import pandas as pd
t = time.time()
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
ctry = dict(zip(s1.entity_id.values, s1.country.values))
runs = {}
for d in sorted(glob.glob("output*/matching_results.tsv")):
    if "sample" in d: continue
    name = d.split("/")[0].replace("output_", "").replace("output", "v1")
    lists = {}
    with open(d, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            a = line.rstrip("\n").split("\t"); lists[a[0]] = frozenset(x for x in (a[1] if len(a) > 1 else "").split(",") if x)
    runs[name] = lists
print(f"loaded {len(runs)} runs in {time.time()-t:.0f}s", flush=True)
ents = list(runs["v7fr_A"].keys()); cc = [ctry[e] for e in ents]
# per-country ids/entity and empty share
print("\nrun            " + "".join(f"{c:>14s}" for c in ("US", "India", "France")) + "   (ids/entity | empty%)")
for name, L in runs.items():
    n = collections.Counter(); tot = collections.Counter(); emp = collections.Counter()
    for e, c in zip(ents, cc):
        v = L.get(e, frozenset()); n[c] += len(v); tot[c] += 1; emp[c] += (not v)
    print(f"{name:14s}" + "".join(f"{n[c]/tot[c]:8.3f}|{100*emp[c]/tot[c]:5.2f}" for c in ("US", "India", "France")))
# entity-level disagreement vs v7fr_A, per country
ref = runs["v7fr_A"]
print("\nentities differing from v7fr_A (%):     US    India   France   | pairs +added -removed vs v7fr_A")
for name, L in runs.items():
    if name == "v7fr_A": continue
    diff = collections.Counter(); tot = collections.Counter(); add = rem = 0
    for e, c in zip(ents, cc):
        a, b = L.get(e, frozenset()), ref[e]; tot[c] += 1
        if a != b: diff[c] += 1; add += len(a - b); rem += len(b - a)
    print(f"  {name:14s} " + "".join(f"{100*diff[c]/tot[c]:8.3f}" for c in ("US", "India", "France")) + f"   | +{add:,} -{rem:,}")
# consensus among the strong (0.984-class) runs
strong = [r for r in ("v6ens_src", "v6full", "v7", "v7fr", "v7fr_A", "hybrid", "vote3", "fr_0.55") if r in runs]
cnt = collections.Counter()
for r in strong:
    for e, v in runs[r].items():
        for p in v: cnt[(e, p)] += 1
k = len(strong)
hist = collections.Counter(cnt.values())
print(f"\nconsensus over {k} strong runs {strong}: pairs by #runs voting: " + ", ".join(f"{i}:{hist[i]:,}" for i in sorted(hist)))
inall = sum(1 for v in cnt.values() if v == k); inref = len({(e, p) for e, v in ref.items() for p in v})
print(f"  pairs in ALL strong runs {inall:,}; pairs in v7fr_A {inref:,}; v7fr_A pairs not unanimous {inref - sum(1 for (e,p),v in cnt.items() if v==k and p in ref[e]):,}")
byc = collections.Counter()
for (e, p), v in cnt.items():
    if v < k: byc[ctry[e]] += 1
print(f"  contested pairs (not unanimous) by country: {dict(byc)}")
print("ANALYZE_DONE", flush=True)
