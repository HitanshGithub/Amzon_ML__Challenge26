# Amazon ML Challenge 2026 — Business Entity Resolution

Team **COMPUTATION_ISSUE** (Hitansh Jain, Utkarsh Pise, Paras Rana — IIIT Nagpur).
Task: for every Source 1 business record, find all matching Source 2 / Source 3 records across US, India
and (test-only) France; scored by macro F0.5. Public leaderboard best: **0.984044** (run `v6ens_src`).

## Repository layout

| Path | What it is |
|---|---|
| `code/business_entity_resolution/src/` | the pipeline: normalisation, joint TF-IDF blocking + empty-address rescue, two-stage LightGBM, fine-tuned `multilingual-e5-base` cross-encoders, stage-3/4 stacking, one-owner assignment, decision rules; `src/v8/` holds the later reranker/corrector experiments and the measurement scripts |
| `code/business_entity_resolution/aws/` | the exact EC2 run scripts of every version (`v6_cpu.sh` + `v5_*` produced the submitted file) |
| `code/business_entity_resolution/README.md` | end-to-end reproduction, including the exact command sequence of the submitted run |
| `code/business_entity_resolution/requirements.txt` | pinned dependencies (Python 3.12) |
| `Documentation_template.md` | methodology write-up: data analysis, blocking, models, results and error analysis |
| `output_v6ens_src/matching_results.tsv` | the submitted predictions; the 1.45 GB `candidate_pairs.tsv` is only in the submission zip |
| `utils/validate_submission.py` | the challenge's format validator |

The dataset (~2.4 GB of TSVs) is not committed; place it under `dataset/train/` and `dataset/test/`.
All models used are MIT/Apache-2.0 and far below 8B parameters (e5-base ≈ 278M); no external data,
APIs or lookups were used.

## Results (public leaderboard)

| Version | Change | Held-out F0.5 | Board |
|---|---|---|---|
| v1 | separate name/address blocking, one LightGBM | 0.9672 | 0.9597 |
| v2 | joint blocking, two-stage LightGBM | 0.9828 | 0.9735 |
| v3 | + cross-encoder + cluster-consistency features | 0.9883 | 0.9835 |
| **v6ens_src** | decoy-simulated training, CE v2+v3 ensemble, 3 seeds, per-source thresholds | 0.9889 | **0.9840** |
| v7fr_A | 60% training entities, French-augmented CE, rule-A filter | 0.9893 | 0.984 |
| v8 | group-trained reranker + uncertain-band corrector | 0.9894 | 0.9828 |
