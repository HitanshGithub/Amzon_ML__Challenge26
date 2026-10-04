"""Compare a new matching_results.tsv with previous submissions: per country, how many entities
changed, ids added / removed, and the empty-list rate. Run from student_resource/."""
import sys, csv, collections, pandas as pd
new = sys.argv[1]; olds = sys.argv[2:]
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
ctry = dict(zip(s1.entity_id.values, s1.country.values))
def load(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            a = line.rstrip("\n").split("\t"); out[a[0]] = set(x for x in (a[1] if len(a) > 1 else "").split(",") if x)
    return out
N = load(new)
for old in olds:
    O = load(old); ch = collections.Counter(); add = collections.Counter(); rem = collections.Counter(); n = collections.Counter()
    e_new = collections.Counter(); e_old = collections.Counter()
    for k, v in N.items():
        c = ctry.get(k, "?"); o = O.get(k, set()); n[c] += 1
        if v != o: ch[c] += 1
        add[c] += len(v - o); rem[c] += len(o - v); e_new[c] += (not v); e_old[c] += (not o)
    print(f"\n{new}  vs  {old}")
    for c in n:
        print(f"  {c:8s} entities {n[c]:8d} changed {ch[c]:7d} ({ch[c]/n[c]:.3%})  +ids {add[c]:7d}  -ids {rem[c]:7d}  empty new {e_new[c]/n[c]:.4f} old {e_old[c]/n[c]:.4f}")
