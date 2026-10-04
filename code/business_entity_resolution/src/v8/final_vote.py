"""Final: vote6 minus pairs the 0.98276 submission added relative to the 0.984 submission."""
def load(p):
    d, order = {}, []
    with open(p, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            a = line.rstrip("\n").split("\t"); order.append(a[0]); d[a[0]] = [x for x in (a[1] if len(a) > 1 else "").split(",") if x]
    return d, order
A, order = load("output_v7fr_A/matching_results.tsv"); H, _ = load("output_v8_hybrid/matching_results.tsv"); V, _ = load("output_vote6/matching_results.tsv")
import os; os.makedirs("output_final_vote", exist_ok=True)
removed = 0
with open("output_final_vote/matching_results.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write("source1_entity_id\tmatched_entity_ids\n")
    for e in order:
        a, h = set(A[e]), set(H[e])
        keep = [p for p in V[e] if not (p in h and p not in a)]
        removed += len(V[e]) - len(keep)
        fo.write(e + "\t" + ",".join(sorted(keep)) + "\n")
print("removed suspect pairs:", removed)
