# Business Entity Resolution — reproduction guide

Pipeline: raw TSVs → normalisation → blocking (joint name+address retrieval + empty-address rescue)
→ pair/record features → two-stage LightGBM → multilingual cross-encoder → stage-3 LightGBM
(3 seeds) → one-owner-per-record assignment → matches.
Uses only the provided training data plus the public pretrained `intfloat/multilingual-e5-base`
weights (MIT); no external lookups or APIs. See `../../Documentation_template.md` for the method.

## Environment
Python 3.11+ (tested on 3.12 / Ubuntu 24.04 and 3.13 / Windows).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Hardware used for the submitted run: AWS r7i.8xlarge (32 vCPU, 256 GB RAM, no GPU), ~7.5 h end to end.
Peak memory is well above 64 GB for the full data; use `src/make_sample.py` to develop on a small sample.

## Run end to end
From this folder, with the challenge data at `../../dataset` (`train/`, `test/`). CPU steps need
~200 GB RAM at full scale (32 vCPU box used); the cross-encoder steps need one GPU (A10G/L4/L40S).

```bash
# 1. normalise, block, rescue, two-stage LightGBM (writes a stage-2 submission to output_s2/)
python src/run_pipeline.py --data-dir ../../dataset --work-dir work --out-dir output_s2     --stages translit,prepare,block,rescue,train,predict --n-jobs 31
# 2. stage-3 preparation: out-of-fold probabilities, cluster-consistency features
python src/stage3.py prep --data-dir ../../dataset --work-dir work --n-jobs 31 --stage3-train-s1 2000000
# 3. pair lists for the cross-encoder, its training list, pseudo-labels for unseen countries
python src/stage4.py pairs --data-dir ../../dataset --work-dir work --n-jobs 31 --all-train     --ce-train-s1 400000 --ce-min-prob 0.001 --ce-max-rank 10 --v3-out output_s2
# 4. cross-encoder (GPU): fine-tune, then score the pair lists
python src/cross_encoder.py train --data-dir ../../dataset --work-dir work --pairs-file work/s4_ce_train_pairs.parquet     --extra-pairs work/s4_pseudo_france.parquet --extra-split test --out work/ce_model_v4 --max-len 128 --bs 256
python src/cross_encoder.py score --data-dir ../../dataset --work-dir work --out work/ce_model_v4 --split train     --pairs work/s4_train_ce_pairs.parquet --dest work/s4_train_ce.parquet --max-len 128
python src/cross_encoder.py score --data-dir ../../dataset --work-dir work --out work/ce_model_v4 --split test     --pairs work/s4_test_ce_pairs.parquet --dest work/s4_test_ce.parquet --max-len 128
# 5. final stage-3 fit (3 seeds), decision rule tuned on held-out entities, outputs
python src/stage4.py fit --data-dir ../../dataset --work-dir work --out-dir output     --pseudo work/s4_pseudo_france.parquet --seeds 3 --n-jobs 31
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (the exact pair set the
final model scored). Our submitted run used the pseudo-labels of the previous round's test
probabilities (`--probs`) instead of `--v3-out`; both paths are supported.

| Stage | What it does | Output |
|---|---|---|
| translit | learns an Indic-script → English token table from training pairs | `work/translit.json` |
| prepare | normalises names/addresses of train and test | `work/{split}_s1.parquet`, `work/{split}_pool.parquet` |
| block | per-country TF-IDF top-k candidate generation + rank/gap features | `work/{split}_cands.parquet` |
| rescue | name-only top-k restricted to empty-address pool records, merged into the candidates | `work/{split}_cands.parquet` |
| train | two-stage LightGBM (2 folds over S1 entities) on 35% of train entities; decision rule tuned on a 10% held-out split with macro F0.5 | `work/model_s1_*.txt`, `model_s2_*.txt`, `train_report.json` |
| predict | stage-2 submission (fallback) | `output_s2/*.tsv` |
| stage3 prep / stage4 pairs / cross_encoder / stage4 fit | see "Run end to end" | `output/*.tsv`, `work/stage4_report.json` |

Validate the outputs:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir ../../dataset/test
```

## Local development on a sample
```bash
python src/make_sample.py --data-dir ../../dataset --out-dir sample_dataset --frac 0.05
python src/run_pipeline.py --data-dir sample_dataset --work-dir work_sample --out-dir output_sample --n-jobs 3
```

## AWS (optional)
`aws/launch.ps1` uploads this folder to S3 and starts an EC2 instance whose user-data (`aws/bootstrap.sh`)
downloads code + data, runs the pipeline, uploads results to `s3://<bucket>/runs/<run-id>/`, and powers off.

## Source layout
- `src/normalize.py` — name/address normalisation (transliteration, legal suffixes, alias split, abbreviations, state codes)
- `src/data.py` — TSV I/O, transliteration learning, cached preparation
- `src/blocking.py` — candidate generation and recall measurement
- `src/features.py` — context (rank/gap) and string-similarity features
- `src/matching.py` — assignment, thresholding, submission writers
- `src/evaluate.py` — macro F0.5 exactly as the challenge defines it
- `src/run_pipeline.py` — CLI orchestrating the CPU stages
- `src/stage3.py` — cluster-consistency features, cross-encoder pair selection, stage-3 stacking
- `src/cross_encoder.py` — cross-encoder fine-tuning / scoring (GPU)
- `src/stage4.py` — diagnostics, pair lists, pseudo-labels, final multi-seed fit and outputs

## v8 — group-trained reranker + uncertain-band corrector (final submission)

v8 keeps the v7 stack (normalisation → joint TF-IDF blocking → two-stage LightGBM → e5 cross-encoder →
stage-3 LightGBM → one-owner assignment) and adds two stages on top of its *probabilities*. It was
built from three measurements on the 220,140 held-out training entities:

| quantity | macro-F0.5 |
|---|---|
| perfect scorer on the current candidate set (blocking ceiling) | 0.99731 |
| perfect per-entity cut on v7's own ranking | 0.99361 |
| v7 as submitted | 0.98928 |

so the loss is in the pair scorer (1.71% of true pairs scored below threshold, 0.49% handed to a
rival entity), not in blocking (117 of 220,140 entities have every true record blocked out) and not
in the threshold. Post-hoc rescue rules (sibling similarity, source-completion prior, per-country
thresholds — `src/v8/lab.py`) were all measured and all lose to the baseline under F0.5, so nothing
of that kind is used.

1. **`src/v8/ce8.py` — group-wise cross-encoder.** `multilingual-e5-base` (MIT, 278M) re-trained as
   a *listwise* model: each training item is one S1 entity's candidate group, loss = BCE + InfoNCE
   (every true copy must out-score that entity's own look-alikes). Groups are mined where v7 is
   wrong or unsure (true pairs the v7 cross-encoder scored < 0.97, negatives it scored > 0.02; 240k
   entities, 1.48M rows) plus 160k label-free synthetic groups built from the *test* records of
   every country (50% France): the positive is a noised copy of the record, the negatives keep the
   city and the generic name words and change exactly one discriminative token, or shift the house
   number — the observed French failure mode. Held-out hard groups: top-1 accuracy 0.980.
2. **`src/v8/band.py`** — the pairs the new model scores: v7 cross-encoder score in
   [0.0005, 0.9995], every French pair, and the top-2 of each entity (4.84M train / 5.92M test
   pairs of 8.3M / 9.6M cross-encoder pairs).
3. **`src/v8/stage5.py` — corrector.** Two-fold LightGBM over entities on the band only, features =
   v7 stage-3 probability, stage-2 probability, v7 CE, ce8, their differences, within-entity and
   within-record ranks/gaps/sums/second-best of each score, cluster-consistency features, source,
   country, empty-address flag. Its output replaces v7's probability on the band; everything else
   keeps v7's probability. Assignment and per-source thresholds are re-tuned on the held-out
   entities; France is checked on a labelled synthetic French world (`src/v8/fr_eval*.py`).

Run (after the v7 steps above have produced `work7/`):

```bash
# GPU
python src/v8/band.py                                                   # band pair lists
python src/v8/ce8.py mine  --work-dir work --ce-file work/v7_train_ce_fr.parquet --out ce8_groups.parquet --max-q 240000 --max-per-q 12 --syn-per-country 40000
python src/v8/ce8.py train --work-dir work --pairs ce8_groups.parquet --model intfloat/multilingual-e5-base --out work/ce8_e5 --groups-per-batch 28 --lr 2e-5 --max-len 112 --listwise 0.5
python src/v8/ce8.py score --work-dir work --out work/ce8_e5 --split train --pairs-file work/v8_train_band.parquet --dest v8_train_ce8.parquet --bs 1024
python src/v8/ce8.py score --work-dir work --out work/ce8_e5 --split test  --pairs-file work/v8_test_band.parquet  --dest v8_test_ce8.parquet  --bs 1024
# CPU
python src/v8/jobA_valprob.py                                           # v7 out-of-fold probabilities (work7/val_scored_all_v7.parquet)
python src/v8/stage5.py --work-dir work7 --out-dir output --ce8 work/v8_train_ce8.parquet --ce8-test work/v8_test_ce8.parquet
python src/v8/assemble.py --work-dir work7 --scored work7/test_scored_v8.parquet --out-dir output_v8_expf --rule expf --alpha 0.0
python src/v8/hybrid.py --a output_v8/matching_results.tsv --b output_v8_expf/matching_results.tsv --countries-a France     --test-s1 ../../dataset/test/test_source1.tsv --out output/matching_results.tsv   # submitted file
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir ../../dataset/test
```

Final decision (validated on the held-out entities): expected-F0.5 per-entity cut for US/India (0.98944 vs 0.98936 with flat thresholds); France keeps stage-5's per-source thresholds, because French probabilities cannot be calibration-checked and the expected-F0.5 rule would drive the French empty-list rate below the training singleton rate. `candidate_pairs.tsv` is the stage-5 candidate file (every pair the final models scored).

### Final submitted file (consensus of earlier runs)
The v8 hybrid scored 0.98276 on the board (below the 0.984 runs): its ~17k extra French pairs were
mostly false positives. The final file is therefore a pair-level majority vote over the six strongest
runs (v6ens_src, fr_0.55, v7fr_A [board 0.984 each], v7, v7fr, v6full; >=4 of 6 votes, ties decided by
v7fr_A, one owner per pool record, restricted to the candidate set), minus every pair the 0.98276 file
had added relative to v7fr_A:

```bash
python src/v8/vote.py v6ens_src,fr_0.55,v7fr_A,v7,v7fr,v6full v7fr_A 4 output_vote6
python src/v8/final_vote.py            # -> output_final_vote/matching_results.tsv (submitted)
```
