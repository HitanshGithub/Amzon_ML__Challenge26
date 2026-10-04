#!/bin/bash
# v3 on the CPU box, after v2 has finished there:
#   1. France-specific normalisation: re-prepare test, re-block France only (+ empty-address rescue on test)
#   2. empty-address rescue on train
#   3. retrain the two-stage LightGBM on the enlarged candidates -> fallback submission output_v25/
#   4. stage-3 prep (OOF probs, support features, cross-encoder pair lists) -> S3
#   5. wait for cross-encoder scores from the GPU box, stage-3 fit + predict -> output_v3/
B=er-challenge-656791094360; RUN=__RUN__
cd /opt/er; source venv/bin/activate
N=$(( $(nproc) - 1 ))
LOG=/opt/er/v3.log
aws s3 sync s3://$B/code/business_entity_resolution code --delete --only-show-errors
mkdir -p v2_backup && cp -n work/train_report.json work/model_s*.txt v2_backup/ 2>/dev/null; cp -rn output v2_backup/ 2>/dev/null
P="python -W ignore code/src/run_pipeline.py --data-dir dataset --work-dir work --n-jobs $N"
$P --stages prepare,block --splits test --block-countries france
$P --stages rescue --splits train
$P --out-dir output_v25 --stages train,predict
python code/utils/validate_submission.py --matching output_v25/matching_results.tsv --candidate output_v25/candidate_pairs.tsv --test-dir dataset/test
aws s3 sync output_v25 s3://$B/runs/$RUN/output_v25 --only-show-errors
aws s3 cp work/train_report.json s3://$B/runs/$RUN/train_report_v25.json --only-show-errors
echo "V25_DONE"
python -W ignore code/src/stage3.py prep --data-dir dataset --work-dir work --n-jobs $N
for f in s3_train_ce_pairs.parquet s3_test_ce_pairs.parquet; do aws s3 cp work/$f s3://$B/v3/$f --only-show-errors; done
echo "PAIRS_UPLOADED"
until aws s3 ls s3://$B/v3/test_ce.parquet > /dev/null 2>&1; do sleep 60; done
aws s3 cp s3://$B/v3/train_ce.parquet work/train_ce.parquet --only-show-errors
aws s3 cp s3://$B/v3/test_ce.parquet work/test_ce.parquet --only-show-errors
rm -rf output_v3 && python -W ignore code/src/stage3.py fit --data-dir dataset --work-dir work --out-dir output_v3 --n-jobs $N
echo "FIT_EXIT=$?"
python code/utils/validate_submission.py --matching output_v3/matching_results.tsv --candidate output_v3/candidate_pairs.tsv --test-dir dataset/test
aws s3 sync output_v3 s3://$B/runs/$RUN/output_v3 --only-show-errors
aws s3 cp work/stage3_report.json s3://$B/runs/$RUN/stage3_report.json --only-show-errors
echo "RUN_FINISHED"
