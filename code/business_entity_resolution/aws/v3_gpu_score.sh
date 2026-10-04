#!/bin/bash
# v3 GPU box, part 2: score the pairs selected by stage3.py prep with the fine-tuned cross-encoder.
B=er-challenge-656791094360; RUN=__RUN__
cd /opt/er; source venv/bin/activate
aws s3 sync s3://$B/dataset/test dataset/test --only-show-errors
until aws s3 ls s3://$B/v3/s3_test_ce_pairs.parquet > /dev/null 2>&1; do sleep 60; done
for f in s3_train_ce_pairs.parquet s3_test_ce_pairs.parquet; do aws s3 cp s3://$B/v3/$f work/$f --only-show-errors; done
python -W ignore code/src/cross_encoder.py score --data-dir dataset --work-dir work --out work/ce_model --split train \
  --pairs work/s3_train_ce_pairs.parquet --dest work/train_ce.parquet --bs 1024
python -W ignore code/src/cross_encoder.py score --data-dir dataset --work-dir work --out work/ce_model --split test \
  --pairs work/s3_test_ce_pairs.parquet --dest work/test_ce.parquet --bs 1024
aws s3 cp work/train_ce.parquet s3://$B/v3/train_ce.parquet --only-show-errors
aws s3 cp work/test_ce.parquet s3://$B/v3/test_ce.parquet --only-show-errors
echo "RUN_FINISHED"
