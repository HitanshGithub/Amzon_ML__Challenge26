# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** COMPUTATION_ISSUE
**Team Members:** Hitansh Jain, Utkarsh Pise, Paras Rana (IIIT Nagpur)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 records to Source 1 entities with a four-stage pipeline: (1) country-partitioned candidate generation with a *joint* name+address sparse TF-IDF search plus an empty-address rescue search (99.1% recall ceiling at 64 candidates per entity), (2) a two-stage LightGBM pair scorer whose second stage sees *competition* features (how each pair ranks against every other pair claiming the same record or the same entity), (3) a fine-tuned multilingual cross-encoder (`intfloat/multilingual-e5-base`, MIT, 278M parameters) that reads both records as raw text and is stacked into a third LightGBM stage together with cluster-consistency features, and (4) a one-owner-per-record assignment with a decision rule tuned for macro-F0.5. Two data findings drove the largest gains: decoy records practically never share an exact address with a Source 1 entity and practically never have an empty address, while 2% of true matches carry a synthetic replacement name that is entirely out of the Source 1 vocabulary — record-level vocabulary and address-uniqueness features encode this. France, absent from training, is handled by language-agnostic normalisation, French-specific address rules, and self-training on high-confidence test predictions.

---

## 2. Methodology

### 2.1 Problem Analysis

Data: train S1 2.21M entities, S2 5.03M, S3 5.29M (US, India); test S1 1.73M, S2 4.89M, S3 5.08M (US, India, France 15%). Ground truth: 7.64M matched pairs, mean 3.46 matches per entity (1.67 from S2, 1.79 from S3), 5.6% singletons; every S2/S3 record belongs to at most one S1 entity; matches never cross countries.

Noise patterns measured on the training pairs:

- **Names.** ~10% ALL-CAPS, 6% all-lowercase, 8% bracketed junk (`[LLC]`, `(ID: 93967)`), 1.6% leading junk (`--`, `...`, `<<`), 4% domain-style names (`coastaltungsten.com`), 2% leet substitutions (`C0astal`, `5érvices`, `lnc`), honorific/filler insertions (`Sri`, `Smt`, `Mr`, `Dr`, `The`: ~100% only on the S2/S3 side), generic-word insertions (`Center`, `Services`, `Partners`, `Enterprises`), word-level abbreviations (`pr`→private, `syst`→systems, `bros`→brothers), legal-form drift (Pvt Ltd / Private Limited / Limited), alias clauses (`Avixylo dba Gonzalez Landscaping`, `f/k/a`, `formerly:`), word-order shuffles, typos, and **synthetic replacement names** (`Viodeltamira`, `Nylazeph`, `ONYXHALO`) in 2% of true matches — 88% of those consist entirely of tokens that never occur in any Source 1 name.
- **Scripts.** 15–19% of S2/S3 names and 9–15% of addresses use Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Odia or Gurmukhi script; 0.4% contain zero-width characters.
- **Addresses.** 3.3% empty; abbreviations (`Rd`/`Road`, `R.`/`Rue`, `Bd`), state written as code or name (`MH`/`Maharashtra`, `KY`/`Kentucky`), component reordering, zero-padded and truncated house numbers (`00412`, `1014`→`014`), ±1 house-number perturbations, spelled-out ordinals, `NULL`/`N/A`/`N°` tokens, landmark references (5% `near/opp`), typos (`stret`, `rod`, `flor`).
- **Decoys.** 26% of the training pool (an estimated 40% of the test pool) matches nothing. Decoys share an exact normalised address with a Source 1 entity in <0.05% of cases (true copies: 27%), have an empty address in 0.3% of cases (true copies: 4.4%), and share an exact name with an S1 entity in 10.8% of cases (true copies: 64%). The test pool's empty-address rate (2.67% ≈ 4.4% × 60% matched) confirms that test decoys are unrelated businesses rather than copies of removed entities.
- **France (test only).** S1 addresses end with the *region* (`Hauts-de-France`) while S2/S3 use the *department* (`Nord`); `St`/`Ste` mean Saint/Sainte, not Street; frequent abbreviations `R.`, `Av`, `Bd`, `All.`, `Imp.`, `Crs`, `Chem.`, `Nº`, `bis/ter`; legal forms SARL/SAS/SASU/EURL/SA/SCI/EI; `Établissements`, `Sté`, `Cie`, `Ets`.

### 2.2 Solution Strategy

**Approach Type:** Blocking + stacked classifiers (sparse retrieval → LightGBM → multilingual cross-encoder → LightGBM) with graph-consistency features and one-to-one assignment.

**Core Innovation:** (a) joint name+address sparse retrieval over the whole pool instead of separate name/address top-k lists (recall ceiling 92.5%/96.0% → 97.8%/98.4% India/US at fewer candidates), plus a name-only rescue restricted to empty-address records (→ 99.1%); (b) a competition-aware second stage (rank/gap of each pair among all pairs claiming the same record and the same entity, second-best competitor scores); (c) stacking a fine-tuned multilingual cross-encoder with cluster-consistency ("do this record and the entity's other strong candidates resemble each other, across sources?") and record-level vocabulary/uniqueness features; (d) transliteration tables for nine Indic scripts learned from the training pairs themselves; (e) self-training for the unseen country.

### 2.3 v8: where the remaining loss is, and the two stages that attack it

Three ceilings were measured on the 220,140 held-out training entities before changing anything:

| macro-F0.5 on the held-out entities | |
|---|---|
| perfect scorer on the existing candidate set (blocking ceiling) | 0.99731 |
| perfect per-entity prefix cut on v7's own ranking | 0.99361 |
| v7 as submitted (P 0.9979, R 0.9714) | 0.98928 |

Blocking is therefore not the constraint (only 117 of 220,140 held-out entities have every true record outside the candidate set) and the threshold is not either: the per-entity hindsight cut gains 0.0043 and everything below it is the ordering of pairs *within* an entity's list. Every post-hoc rescue we could think of was measured on the same split and lost under F0.5 — accepting a rejected record that is near-identical to an accepted sibling (hit-rate 0.19–0.35 at any similarity), adding the best missing-source record given the P(≥1 S3 | ≥1 S2) = 0.9255 prior (hit-rate 0.61 at best, break-even 0.67), per-country thresholds (+0.00002). The scorer had to get better, so v8 adds:

**Group-wise cross-encoder (`ce8`).** `multilingual-e5-base` re-trained with each S1 entity's candidate list as one training item and loss = BCE + InfoNCE (each true copy must out-score that entity's own look-alikes), i.e. the objective the one-owner assignment actually needs. Groups are mined where v7 is wrong or unsure (true pairs the v7 cross-encoder scored < 0.97, non-matches it scored > 0.02, 240k entities, 1.48M rows) and augmented with 160k label-free synthetic groups built from the *test* records of all three countries (half France). The synthetic negatives keep the city and the generic name words and change exactly one discriminative token or shift the house number, which is what French look-alikes do ("Association du Archives" vs "Association des Arts", "9 Rue X" vs "11 Rue X"); positives apply the observed French noise (legal-form drift SARL/S.A.R.L./[SARL], `R.`/`AV`/`BD`, region↔department swap, wrong accents `Àmicale`, `(France)` insertions, `9 bis`→`9 B`, `NO. 5`, typos, case). Held-out hard groups: top-1 accuracy 0.980.

**Uncertain-band corrector (stage 5).** ce8 is scored only where v7 is not already sure (v7 CE in [0.0005, 0.9995], every French pair, the top-2 of each entity: 4.84M train / 5.92M test pairs). A two-fold LightGBM over entities, trained on that band only, sees v7's stage-3 and stage-2 probabilities, v7's CE, ce8, their differences, within-entity and within-record ranks/gaps/sums/second-best of each score, the cluster-consistency features, source, country and the empty-address flag; its probability replaces v7's on the band and everything else keeps v7's. Assignment and per-source thresholds are re-tuned on the held-out entities. Because France has no labels, it is measured on a labelled synthetic French world (real French test S1 records, copies generated with the observed noise, real French decoys the model is confident match nothing): v7 scores 0.9408 macro-F0.5 at the v7 thresholds (P 0.995, R 0.845) there and v8 0.9445 (P 0.995, R 0.860).

**What the first group-trained model did and did not do.** Trained from the base `multilingual-e5-base` weights on the hard-mined + synthetic mix, it reached 98.0% top-1 on held-out *hard* groups but proved weaker than v7's three-model cross-encoder ensemble on the real uncertain band (AUC 0.968 vs 0.993; on the rows where the old ensemble is itself unsure, 0.66 vs 0.82, against 0.94 for v7's stacked probability): capacity spent on the skewed mining distribution cost calibration on ordinary pairs, and the stage-5 corrector accordingly leaned on v7's probability (validation 0.98930 → 0.98936; France synthetic world 0.9408 → 0.9445 at the same thresholds). The second model (`ce8-v2`) therefore *continues* v7's French-augmented cross-encoder weights, keeps its plain `name | address` input, and trains on full real candidate groups of 356k entities (2.0M rows) with the hard groups folded in and the synthetic groups reduced to a 12% minority, at a lower learning rate; it is far better calibrated than the first (held-out hard groups: top-1 0.989, positives below 0.5 2.7%; band AUC 0.9921 vs 0.968) but, being one model, still no better than v7's three-model ensemble (0.9933), and the corrector gains nothing from it (0.98936); on the synthetic French world it is *worse* at the operating thresholds (0.9381 vs 0.9445 for the first model), so it is not used.

**Where the remaining loss sits.** Splitting the held-out entities by whether their exact normalised name is shared with another S1 entity of the same country: shared-name entities are ~50% of entities but carry ~77% of the lost score, almost all of it as recall (0.955 vs 0.988 for unique names; precision 0.997 in both groups) — copies of look-alike twins, many with an empty address, that no string or model feature can attribute. Group-specific thresholds recover only +0.00006.

**Final decision rule.** The per-entity expected-F0.5 prefix cut (add a candidate only while the entity's expected F0.5 rises) beats flat per-source thresholds by +0.00008 on the held-out entities and is used for US and India; for France, where probabilities are less extreme and calibration cannot be validated, it would push the empty-list rate (5.0%) below the training singleton rate (5.6%), so France keeps the stage-5 per-source thresholds (0.75/0.75).

Two further ideas were measured and rejected on the held-out split: a conflict-aware re-assignment of records whose top two owners are within δ (swaps 3–40 records, no change), and every post-hoc rescue rule listed above.

### 2.4 Final submission: consensus of the board-tested runs

Submitted, the v8 hybrid scored 0.98276, below the 0.984 runs. Against v7fr_A it had added ~17k French pairs (the new corrector raised French probabilities after training on synthetic French groups), and the board shows they were mostly false positives — the synthetic French world, where adding French pairs helped, does not reproduce real French errors. Board evidence across runs: v6ens_src has 22.6k fewer French pairs than v7fr_A and both score 0.984; fr_0.92 (46k fewer) loses 0.0006; the hybrid (17k more) loses 0.0013 — France is on a plateau between v6 and v7fr_A and falls off when pairs are added. The final file is a pair-level majority vote of the six strongest runs (>=4/6, ties to v7fr_A, one owner per record, inside the candidate set) with every pair the 0.98276 file introduced removed (8,894). It differs from v7fr_A on 0.2% of US/India entities and removes 15.7k French pairs supported only by the v7fr French cross-encoder.

---

## 3. Candidate Generation (Blocking)

- **Partitioning:** by the `country` string (treated as an open set; France gets its own block automatically).
- **Normalisation:** Unicode NFKC + `unidecode` transliteration; a learned token table maps 8,394 non-Latin tokens to their English counterparts (learned by co-occurrence × phonetic similarity over the 1.58M training pairs that contain non-Latin text); alias-clause splitting; legal-form canonicalisation; leet de-substitution; abbreviation expansion; state-code expansion (US, India); French street/legal abbreviations; repeated-letter collapse for transliteration robustness (`raam`→`ram`); zero-padded number stripping; drop of `NULL`/`N/A`/`N°`.
- **Blocking keys used:** one sparse vector per record `[√w·TF-IDF(char 3-grams of name) ‖ √(1−w)·TF-IDF(words of address)]`, w = 0.4, so that a dot product equals `w·cos_name + (1−w)·cos_addr`; top-40 per S1 entity over the whole country pool via `sparse_dot_topn`; character 3-grams present in >5% of names are dropped (same recall, ~20% faster). Plus name-only top-10 (cosine ≥ 0.1) and, as a rescue, name-only top-20 (cosine ≥ 0.3) restricted to the 3.4% of pool records with an empty address — 77% of the misses of the joint search had an empty address.
- **Candidate pairs generated:** test 110,741,895 pairs (63.9 per S1 entity, reduction ratio 0.999994); train 140,333,989 (63.6 per entity).
- **How we ensured true matches were not lost:** recall ceiling measured on all 7,638,365 training pairs at every design change: v1 (separate name top-30 ∪ address top-10) 94.38% → joint search 98.13% → + empty-address rescue **99.11%**. Weight, k and max-df were chosen on 3k–5k query entities against the *full* country pools, because the 5% development sample (99.3%) badly understated look-alike competition.

---

## 4. Matching Model

**Features used (98 per pair):**
- Name features: RapidFuzz ratio / token-sort / token-set / partial ratios on normalised, core (legal forms removed) and squashed (no spaces, matches domain-style names) forms, Jaro-Winkler, token Jaccard, first-token equality, legal-form equality/Jaccard, length difference, abbreviation-aware token matching (prefix and ≥80% similarity), "S1 name contained in candidate", number of extra tokens, TF-IDF cosines from blocking.
- Address features: ratio / token-set / token-sort / Jaccard on normalised addresses, number-set Jaccard, first-number equality, truncated/zero-padded number match, postal-code equality, minimal house-number distance (log), empty-address flags.
- Record-level features: fraction (and all-or-nothing flag) of the candidate's name tokens absent from the Source 1 vocabulary of its country; counts of S1 entities and pool records sharing the exact normalised address / name, for both sides; source (S2/S3); domain-style, non-ASCII and alias flags.
- Competition features (stage 2 and 3): rank and gap of the pair's stage-1/stage-2 probability among all pairs of the same S1 entity and among all pairs claiming the same S2/S3 record, per-source rank, number of strong competitors, sum of probabilities, second-best competitor.
- Cluster-consistency features (stage 3): for each of the entity's top-6 candidates, maximum name/address similarity to the other top-6 candidates, similarity to the best cross-source sibling, number of similar siblings, probability-weighted similarity.
- Cross-encoder score (stage 3): `multilingual-e5-base` fine-tuned as a pair classifier on 6.08M pairs (`name | address` of both records, 128 tokens): all true pairs of 400k training entities, the 6 hardest stage-2 negatives and 2 random negatives per entity, plus 1.5M balanced pseudo-labelled French test pairs. Held-out log-loss 0.0054, precision 0.985 / recall 0.989 at 0.5. Scored on the top-10 candidates per entity with stage-2 probability ≥ 0.001 (99.9% of true pairs).

**Model type:** three stacked LightGBM classifiers (255 leaves, lr 0.05, early stopping), each trained two-fold over S1 entities so that every training pair receives an honest out-of-fold probability for the next stage; stage 3 uses a 3-seed ensemble and includes pseudo-labelled French rows. Final parameter count ≈ 278M (cross-encoder) + tree models; all components MIT/Apache-2.0.

**Threshold selection method:** each S2/S3 record is assigned to its highest-probability S1 entity (100% of training pairs have a single owner); then, per entity, either a global threshold, per-source (S2/S3) thresholds, or the top-j prefix maximising plug-in expected F0.5 is chosen — whichever maximises macro-F0.5 on a 10% held-out set of 220,140 S1 entities never used in training (the same split for all stages). Precision is always ≥ 0.997 on validation; recall is the quantity being traded.

---

## 5. Results & Error Analysis

Validation = macro-F0.5 on the 220,140 held-out training entities (US/India); leaderboard = public test subset (includes France).

| Version | Change | Recall ceiling | Validation F0.5 | Leaderboard |
|---|---|---|---|---|
| v1 | separate name/address blocking, 1 LightGBM | 94.4% | 0.9672 | 0.9597 |
| v2 | joint blocking, 2-stage LightGBM, expected-F decision | 98.1% | 0.9828 | 0.9735 |
| v2.5 | + empty-address rescue, French rules | 99.1% | 0.9818* | 0.9739 |
| v3 | + cross-encoder + cluster features (stage 3), per-source thresholds | 99.1% | 0.9883* | 0.9835 |
| v4 | + cross-encoder v2 (hard negatives, 128 tokens, French pseudo-labels), wider CE coverage, 3 seeds | 99.1% | 0.9888* | — |
| v5 | + record-level vocabulary/uniqueness features in all stages; training set without artificial entity dropping | 99.1% | stage 2: 0.9844; stage 3: **0.9896** (P 0.9987, R 0.970) | see below |
| v5 variants | same pipeline, differing only in the France treatment: `v5np` = no pseudo-labelled rows (0.9897); `v5ce1` = v3-era cross-encoder without French pseudo-pairs (0.9893); `v5ce3` = v2 recipe cross-encoder trained on US/India only (0.9896); `v5ens` = v2+v3 ensemble (0.9898) | 99.1% | 0.9893–0.9898 | v5ce3: **0.9817** |
| **v6** | v3's training recipe (decoy simulation, no record-count features) + cross-encoder v2+v3 ensemble + 3 seeds + all training entities, per-source thresholds | 99.1% | **0.9889** (P 0.9978, R 0.970) | **0.9840** |
| **v8** | v7 probabilities + group-trained cross-encoder scored on the uncertain band + stage-5 corrector; assignment and thresholds re-tuned; France checked on a labelled synthetic French world (0.9408 → 0.9445 at v7 thresholds) | 99.1% | v8-v1 (e5-base from scratch): 0.98936; v8-v2 (continued from v7's CE): 0.98936; **final = v8 corrector + expected-F0.5 cut for US/India: 0.98944** (P 0.998, R 0.972) | 0.98276 |

\* measured with a harder, test-like decoy density (20% of entities removed from the candidate set during training); v5 dropped this after finding it unfaithful (see §2.1).

- **Common false positives (wrong merges):** look-alike businesses with the same street and a house number differing by one digit (`RR Valley Harbors, 701 Hooper Rd` vs `RZR Valley-Harbors, 714 Hooper Rd`); a decoy with a name variant at an ambiguous address; two S1 entities sharing an address (singleton predicted as matched). Validation precision is 0.9975–0.9978, i.e. 0.22–0.25% of predicted pairs.
- **Common false negatives (missed matches):** (i) records outside the candidate set (0.9%): heavy-typo names with no address, generic names; (ii) empty-address copies with a generic name variant (`Zeus`, `Fresh Tattoo Inc.`) — 68% of the remaining recall loss on the development sample, addressed in v5 by the uniqueness counts and the faithful training set; (iii) synthetic replacement names at truncated addresses (`ONYXHALO | G-30, MUMBAI`); (iv) one-digit house-number noise combined with a name variant, which the cross-encoder treats strictly.
- **France decision threshold (leaderboard experiment):** with the v6 probabilities, raising the French threshold from 0.75 to 0.92 (−3.3% French matches) lowered the board score by 0.0006 and lowering it to 0.55 (+2.2%) changed it by −0.00005, so the decision cut is already optimal for France; the remaining French loss is in pair scoring, which cannot be improved without French labels. A zero-shot 7B LLM judge (Qwen2.5-7B-Instruct) was also evaluated on 30k labelled uncertain pairs: AUC 0.60 vs 0.86 for the stage-2 model — the noise is synthetic and not amenable to world knowledge — and was not used.
- **France:** the two scorers disagree three times more often than on US records (0.40 vs 0.13 pairs per entity); most disagreements are synthetic replacement names at the exact S1 address, which the cross-encoder accepts and the string model rejects. The v5 record-count features, which raised validation, *lowered* the board (0.9835 → 0.9817): France's address structure differs — 13.3% of French S1 entities share their exact normalised address with another S1 entity (US/India 4–5%), 47% of French pool records sit at an address belonging to some S1 entity (11–21%), and French names are highly repetitive (52% of French S1 entities share their exact name with another). Address-uniqueness counts learned on US/India therefore re-rank owners among co-located French entities wrongly. Lesson: features whose statistics depend on the pool's composition do not transfer to an unseen country; the final model excludes them.

---

## 6. Conclusion

A sparse joint name+address retrieval with a targeted empty-address rescue gives a 99.1% recall ceiling on 10M-record pools; stacking competition-aware LightGBM stages with a fine-tuned multilingual cross-encoder and cluster-consistency features turns that into ~0.989 macro-F0.5 on held-out entities. The decisive lessons were empirical: measure blocking recall on full-size pools, not samples; interrogate what decoys look like before simulating them; and give the model explicit signals for the noise generator's own artefacts (synthetic names, empty addresses, address uniqueness). Self-training and language-agnostic normalisation carried the system to a country it had never seen.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python 3.12; `requirements.txt` pins numpy, pandas, scipy, scikit-learn, pyarrow, lightgbm, rapidfuzz, sparse-dot-topn, unidecode, torch, transformers).

- `src/normalize.py` — name/address normalisation, transliteration application, alias splitting, legal forms, abbreviations, state codes, French rules.
- `src/data.py` — TSV I/O, transliteration-table learning, cached normalised records.
- `src/blocking.py` — joint TF-IDF candidate generation, empty-address rescue, recall measurement.
- `src/features.py` — pair string features, competition (context) features, record-level vocabulary/uniqueness features.
- `src/model.py` — two-fold LightGBM training, competition features from probabilities, assignment and decision rules.
- `src/run_pipeline.py` — stages `translit, prepare, block, rescue, train, predict`.
- `src/stage3.py` — cluster-consistency features, cross-encoder pair selection, stage-3 stacking.
- `src/cross_encoder.py` — cross-encoder fine-tuning and scoring (GPU).
- `src/stage4.py` — diagnostics, pair lists, pseudo-labels, final multi-seed fit and output writing.
- `src/evaluate.py` — macro-F0.5 exactly as defined by the challenge. `src/make_sample.py` — density-preserving 5% development sample.
- `src/v8/` — v8: `ce8.py` (group-wise cross-encoder: mining, listwise training, scoring, synthetic groups), `band.py` (uncertain-band pair lists), `stage5.py` (band corrector + decision tuning + outputs), `assemble.py` (final decision rule), `jobA_valprob.py` (v7 out-of-fold probabilities), `lab.py` / `fnan.py` / `reassign.py` / `ceiling.py` / `dropval.py` (measured-and-rejected ideas, ceilings), `fr_eval.py` / `fr_eval5.py` (labelled synthetic French world), `compare_outputs.py`.
- `aws/` — EC2 bootstrap/launch scripts (`v8_*` = the v8 chains) (r7i.8xlarge for CPU stages, g5/g6e for the cross-encoder).

Reproduction (from `code/business_entity_resolution`, data at `../../dataset`):

```
python src/run_pipeline.py --data-dir ../../dataset --work-dir work --stages translit,prepare,block,rescue,train --n-jobs 31
python src/stage3.py prep --data-dir ../../dataset --work-dir work --n-jobs 31 --stage3-train-s1 2000000
python src/stage4.py pairs --data-dir ../../dataset --work-dir work --all-train --ce-train-s1 400000 --v3-out output_v3   # or --probs <previous test probabilities>
python src/cross_encoder.py train --data-dir ../../dataset --work-dir work --pairs-file work/s4_ce_train_pairs.parquet --extra-pairs work/s4_pseudo_france.parquet --extra-split test --out work/ce_model_v4 --max-len 128
python src/cross_encoder.py score --data-dir ../../dataset --work-dir work --out work/ce_model_v4 --split train --pairs work/s4_train_ce_pairs.parquet --dest work/s4_train_ce.parquet --max-len 128
python src/cross_encoder.py score --data-dir ../../dataset --work-dir work --out work/ce_model_v4 --split test  --pairs work/s4_test_ce_pairs.parquet  --dest work/s4_test_ce.parquet  --max-len 128
python src/stage4.py fit --data-dir ../../dataset --work-dir work --out-dir output --pseudo work/s4_pseudo_france.parquet --seeds 3
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir ../../dataset/test
```

`output/candidate_pairs.tsv` is exactly the set scored by the final model (110.7M pairs); `output/matching_results.tsv` is its output.

### B. Additional Results

- Blocking benchmark (3k–5k queries against full country pools, recall of true pairs): name top-30 ∪ address top-10 = 92.5% (India) / 96.0% (US); joint w=0.5 top-30 = 96.7% / 97.7%; joint w=0.4 top-40 ∪ name top-10 = 97.8% / 98.4%; + empty-address rescue top-20 = 98.8% / 99.3%.
- Validation loss decomposition (v3, 760,567 true pairs of the held-out entities): 0.90% not in candidates, 0.49% assigned to a competing entity, 1.71% below the decision threshold (13% of those never scored by the cross-encoder); false predictions 0.25%.
- Cross-encoder held-out quality: v3 (96 tokens, 3.4M pairs) log-loss 0.0118, precision 0.945 / recall 0.996 at 0.5; v4 (128 tokens, 6.1M pairs incl. French pseudo-pairs, stage-2 hard negatives) log-loss 0.0054, precision 0.985 / recall 0.989.
- Compute: full CPU pipeline ≈ 8 h on 32 vCPU / 256 GB (candidate generation dominates: 4.3 h train, 2.5 h test); cross-encoder fine-tuning 55 min on an L40S (1,800 pairs/s), scoring 15M pairs ≈ 1 h.
