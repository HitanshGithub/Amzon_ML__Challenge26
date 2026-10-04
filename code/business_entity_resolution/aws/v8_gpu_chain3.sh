#!/bin/bash
# ce8-v2: continue v7's French-augmented cross-encoder (ce_model_fr) on full real groups + hard groups
# + a small synthetic minority, listwise loss; plain "name | address" text (the format it was trained on).
set -uxo pipefail
B=er-challenge-656791094360
export HOME=/root PATH=/opt/er/venv/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH
cd /opt/er; source venv/bin/activate; export PYTHONPATH=/opt/er/code/src TOKENIZERS_PARALLELISM=false
until grep -q MINE_DONE /opt/er/v8/mine2.log; do sleep 10; done
until grep -q GPU_CHAIN_DONE /opt/er/v8/gpu_chain.log; do sleep 15; done
python -W ignore v8/ce8v2.py train --work-dir work --pairs ce8v2_groups.parquet --init work/ce_model_fr \
  --out work/ce8_v2 --groups-per-batch 28 --epochs 1 --lr 1e-5 --max-len 112 --bs 640 --listwise 0.3 --no-prefix
echo "V2_TRAINED"
python -W ignore v8/ce8v2.py score --work-dir work --out work/ce8_v2 --split train --pairs-file work/v8_train_band.parquet --dest v8_train_ce8v2.parquet --bs 1024 --max-len 112 --no-prefix
aws s3 cp work/v8_train_ce8v2.parquet s3://$B/v8/v8_train_ce8v2.parquet --only-show-errors; echo "TRAIN_V2_UP"
python -W ignore v8/ce8v2.py score --work-dir work --out work/ce8_v2 --split test --pairs-file work/v8_test_band.parquet --dest v8_test_ce8v2.parquet --bs 1024 --max-len 112 --no-prefix
aws s3 cp work/v8_test_ce8v2.parquet s3://$B/v8/v8_test_ce8v2.parquet --only-show-errors; echo "TEST_V2_UP"
python -W ignore v8/ce8v2.py score --data-dir dataset_syn --work-dir work --out work/ce8_v2 --split test --pairs-file work/syn_ce_pairs.parquet --dest v8_syn_ce8v2.parquet --bs 1024 --max-len 112 --no-prefix
aws s3 cp work/v8_syn_ce8v2.parquet s3://$B/v8/v8_syn_ce8v2.parquet --only-show-errors; echo "SYN_V2_UP"
echo "GPU_CHAIN3_DONE"
