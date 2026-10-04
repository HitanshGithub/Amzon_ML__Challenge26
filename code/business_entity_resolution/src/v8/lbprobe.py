"""Use board results as coarse labels: pairs the 0.98276 file added vs the 0.984 file are suspect."""
import csv, collections, sys
s1 = {}
with open("dataset/test/test_source1.tsv", encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t"); s1[p[0]] = p[3]
def pairs(p):
    s = set()
    with open(p, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            a = line.rstrip("\n").split("\t")
            for x in (a[1] if len(a) > 1 else "").split(","):
                if x: s.add((a[0], x))
    return s
A = pairs("output_v7fr_A/matching_results.tsv"); H = pairs("output_v8_hybrid/matching_results.tsv")
V = pairs("output_vote6/matching_results.tsv"); V3 = pairs("output_vote3sub/matching_results.tsv")
Hbad = H - A; Hgood = A - H
def byc(s): return dict(collections.Counter(s1[e] for e, _ in s))
print("hybrid-only (suspect) pairs:", byc(Hbad)); print("v7fr_A-only vs hybrid:", byc(Hgood))
for name, S in (("vote6", V), ("vote3sub", V3)):
    plus, minus = S - A, A - S
    print(f"{name}: +{len(plus):,} {byc(plus)} | -{len(minus):,} {byc(minus)}")
    print(f"   +pairs that are hybrid-suspect: {byc(plus & Hbad)} | -pairs that hybrid also lacked: {byc(minus & Hgood)}")
