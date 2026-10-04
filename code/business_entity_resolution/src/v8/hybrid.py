"""Final submission file: rows of countries in --countries-a from file A, all other rows from file B.
Used as: A = stage-5 output with per-source thresholds (France), B = expected-F0.5 output (US/India).
    python src/v8/hybrid.py --a output_v8/matching_results.tsv --b output_v8_expf/matching_results.tsv \
        --countries-a France --test-s1 ../../dataset/test/test_source1.tsv --out output/matching_results.tsv"""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--countries-a", default="France"); ap.add_argument("--test-s1", required=True)
ap.add_argument("--out", required=True)
x = ap.parse_args()
ca = set(x.countries_a.split(","))
ctry = {}
with open(x.test_s1, encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t"); ctry[p[0]] = p[3]
def load(p):
    d = {}
    with open(p, encoding="utf-8") as fh:
        hdr = fh.readline()
        for line in fh:
            a = line.rstrip("\n").split("\t"); d[a[0]] = a[1] if len(a) > 1 else ""
    return hdr, d
hdr, A = load(x.a); _, B = load(x.b)
os.makedirs(os.path.dirname(x.out) or ".", exist_ok=True)
with open(x.out, "w", encoding="utf-8", newline="") as out:
    out.write(hdr)
    for eid in A:
        out.write(f"{eid}\t{A[eid] if ctry[eid] in ca else B[eid]}\n")
print("hybrid written:", x.out)
