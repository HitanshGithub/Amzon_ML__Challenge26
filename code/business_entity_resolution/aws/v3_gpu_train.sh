#!/bin/bash
# v3 on the GPU box: fine-tune the cross-encoder, then score the pairs the CPU box selected.
B=er-challenge-656791094360; RUN=__RUN__
cd /opt/er; source venv/bin/activate
aws s3 sync s3://$B/code/business_entity_resolution code --delete --only-show-errors
pip install -q -r code/requirements.txt
aws s3 sync s3://$B/dataset/test dataset/test --only-show-errors
nvidia-smi --query-gpu=name,memory.total --format=csv
python -W ignore code/src/cross_encoder.py train --data-dir dataset --work-dir work --cands work/train_cands.parquet \
  --model intfloat/multilingual-e5-base --out work/ce_model --n-s1 300000 --bs 256 --max-len 96 --workers 7
echo "CE_TRAIN_EXIT=$?"
aws s3 sync work/ce_model s3://$B/runs/$RUN/ce_model --only-show-errors
echo "RUN_FINISHED"
