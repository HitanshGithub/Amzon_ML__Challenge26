#!/bin/bash
set -uxo pipefail
B=er-challenge-656791094360
export HOME=/root PATH=/opt/er/venv/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH
cd /opt/er; source venv/bin/activate; export PYTHONPATH=/opt/er/code/src TOKENIZERS_PARALLELISM=false
until grep -q TRAIN_DONE /opt/er/v8/train_e5.log; do sleep 20; done
echo "=== e5 trained ==="
# France measurement world first (small, labelled) so the CPU box can start tuning France early
python -W ignore v8/ce8.py score --data-dir dataset_syn --work-dir work --out work/ce8_e5 --split test \
  --pairs-file work/syn_ce_pairs.parquet --dest v8_syn_ce8.parquet --bs 1024 --max-len 112
aws s3 cp work/v8_syn_ce8.parquet s3://$B/v8/v8_syn_ce8.parquet --only-show-errors
echo "SYN_UP"
python -W ignore v8/ce8.py score --work-dir work --out work/ce8_e5 --split train \
  --pairs-file work/v8_train_band.parquet --dest v8_train_ce8.parquet --bs 1024 --max-len 112
aws s3 cp work/v8_train_ce8.parquet s3://$B/v8/v8_train_ce8.parquet --only-show-errors
echo "TRAIN_BAND_UP"
python -W ignore v8/ce8.py score --work-dir work --out work/ce8_e5 --split test \
  --pairs-file work/v8_test_band.parquet --dest v8_test_ce8.parquet --bs 1024 --max-len 112
aws s3 cp work/v8_test_ce8.parquet s3://$B/v8/v8_test_ce8.parquet --only-show-errors
echo "TEST_BAND_UP"
echo "GPU_CHAIN_DONE"
