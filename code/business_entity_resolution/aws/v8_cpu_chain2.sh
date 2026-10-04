#!/bin/bash
set -uxo pipefail
B=er-challenge-656791094360
export HOME=/root PATH=/opt/er/venv/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH AWS_DEFAULT_REGION=us-east-1
cd /opt/er; source venv/bin/activate
aws s3 cp s3://$B/v8/stage5.py v8/stage5.py --only-show-errors
until aws s3 ls s3://$B/v8/v8_train_ce8v2.parquet >/dev/null 2>&1; do sleep 20; done
aws s3 cp s3://$B/v8/v8_train_ce8v2.parquet work/v8_train_ce8v2.parquet --only-show-errors
echo "=== TRAIN CE8 PULLED $(date -u +%T) ==="
# 1) validation-only fit: the number we steer by
python -W ignore v8/stage5.py --work-dir work7 --ce8 work/v8_train_ce8v2.parquet --ce8-test work/v8_test_ce8v2.parquet --tag v8v2val --no-test 2>&1 | grep -vE "^\[LightGBM\]"
echo "=== S5 VAL DONE $(date -u +%T) ==="
aws s3 cp work7/stage5_report_v8v2val.json s3://$B/v8/stage5_report_v8v2val.json --only-show-errors
# 2) France on the labelled synthetic world with the freshly fitted corrector
until aws s3 ls s3://$B/v8/v8_syn_ce8v2.parquet >/dev/null 2>&1; do sleep 20; done
aws s3 cp s3://$B/v8/v8_syn_ce8v2.parquet work_syn/v8_syn_ce8v2.parquet --only-show-errors
python -W ignore v8/fr_eval5.py --work-dir work_syn --models work7 --ce work_syn/syn_ce_ens.parquet --ce8 work_syn/v8_syn_ce8v2.parquet --prob work_syn/syn_scored_v7.parquet --tag v8v2 2>&1 | grep -E "fr5|Traceback|Error"
echo "=== FR5 DONE $(date -u +%T) ==="
until aws s3 ls s3://$B/v8/v8_test_ce8v2.parquet >/dev/null 2>&1; do sleep 20; done
aws s3 cp s3://$B/v8/v8_test_ce8v2.parquet work/v8_test_ce8v2.parquet --only-show-errors
echo "=== TEST CE8 PULLED $(date -u +%T) ==="
# 3) full run: refit (same seed) + test outputs
rm -rf output_v8v2
python -W ignore v8/stage5.py --work-dir work7 --out-dir output_v8v2 --ce8 work/v8_train_ce8v2.parquet --ce8-test work/v8_test_ce8v2.parquet --tag v8v2 2>&1 | grep -vE "^\[LightGBM\]"
echo "=== S5 FULL DONE $(date -u +%T) ==="
python code/utils/validate_submission.py --matching output_v8v2/matching_results.tsv --candidate output_v8v2/candidate_pairs.tsv --test-dir dataset/test
aws s3 cp output_v8v2/matching_results.tsv s3://$B/v8/output_v8v2/matching_results.tsv --only-show-errors
aws s3 cp work7/stage5_report_v8v2.json s3://$B/v8/stage5_report_v8v2.json --only-show-errors
echo "CPU_CHAIN2_DONE"
