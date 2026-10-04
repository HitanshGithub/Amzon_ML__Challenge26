"""Pair-level majority vote over strong submissions, one owner per pool record (most votes, ties to
the reference run), restricted to the final candidate set. Writes several variants for inspection."""
import csv, collections, os, sys
runs = sys.argv[1].split(","); ref = sys.argv[2]; minv = int(sys.argv[3]); out = sys.argv[4]
cand_file = "output_v8/candidate_pairs.tsv"
def load(p):
    d = {}
    with open(p, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            a = line.rstrip("\n").split("\t"); d[a[0]] = [x for x in (a[1] if len(a) > 1 else "").split(",") if x]
    return d
L = {r: load(f"output_{r}/matching_results.tsv") for r in runs}
R = L[ref]
votes = collections.Counter()
for r in runs:
    for e, v in L[r].items():
        for p in v: votes[(e, p)] += 1
refset = {(e, p) for e, v in R.items() for p in v}
keep = {k: n for k, n in votes.items() if n >= minv or (n == minv - 1 and 2 * (minv - 1) == len(runs) and k in refset)}
# one owner per pool record
owner = {}
for (e, p), n in keep.items():
    score = (n, (e, p) in refset)
    if p not in owner or score > owner[p][1]: owner[p] = (e, score)
final = collections.defaultdict(list)
for p, (e, _) in owner.items(): final[e].append(p)
# restrict to candidate set
cand = {}
with open(cand_file, encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        a = line.rstrip("\n").split("\t"); cand[a[0]] = set(x for x in (a[1] if len(a) > 1 else "").split(",") if x)
dropped = 0
os.makedirs(out, exist_ok=True)
s1ids = list(R.keys())
with open(f"{out}/matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write("source1_entity_id\tmatched_entity_ids\n")
    for e in s1ids:
        v = [p for p in final.get(e, []) if p in cand.get(e, ())]
        dropped += len(final.get(e, [])) - len(v)
        fo.write(e + "\t" + ",".join(sorted(v)) + "\n")
tot = sum(len(v) for v in final.values())
print(f"[vote] runs={runs} ref={ref} min_votes={minv}: {tot:,} pairs, dropped-not-in-candidates {dropped:,}; ref has {len(refset):,}", flush=True)
