# v4 GPU job body: wait for stage4 pair lists, fine-tune cross-encoder v2 (stage-2 hard negatives +
# France pseudo-labels, 128 tokens), score the wider pair lists, upload.
B=er-challenge-656791094360; RUN=v4-20260926
cd /opt/er; source venv/bin/activate
aws s3 sync s3://$B/code/business_entity_resolution code --delete --only-show-errors
mkdir -p dataset work
aws s3 sync s3://$B/dataset dataset --only-show-errors
for f in train_s1.parquet train_pool.parquet train_cands.parquet; do [ -f work/$f ] || aws s3 cp s3://$B/v1work/$f work/$f --only-show-errors; done
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: $GPU"
case "$GPU" in *L40S*|*A100*|*H100*) BS=256;; *) BS=160;; esac
until aws s3 ls s3://$B/v4/ 2>/dev/null | grep -q s4_pseudo_; do sleep 60; done; sleep 30
aws s3 sync s3://$B/v4 work --exclude "*" --include "s4_*.parquet" --only-show-errors
EXTRA=$(ls work/s4_pseudo_*.parquet | tr '\n' ',' | sed 's/,$//')
python -W ignore code/src/cross_encoder.py train --data-dir dataset --work-dir work --cands work/train_cands.parquet \
  --model intfloat/multilingual-e5-base --out work/ce_model_v4 --pairs-file work/s4_ce_train_pairs.parquet \
  --extra-pairs "$EXTRA" --extra-split test --bs $BS --max-len 128 --workers 7 --n-val 20000
echo "CE_TRAIN_EXIT=$?"
aws s3 sync work/ce_model_v4 s3://$B/runs/$RUN/ce_model_v4 --only-show-errors
python -W ignore code/src/cross_encoder.py score --data-dir dataset --work-dir work --out work/ce_model_v4 --split train \
  --pairs work/s4_train_ce_pairs.parquet --dest work/s4_train_ce.parquet --bs 1024 --max-len 128
aws s3 cp work/s4_train_ce.parquet s3://$B/v4/s4_train_ce.parquet --only-show-errors
python -W ignore code/src/cross_encoder.py score --data-dir dataset --work-dir work --out work/ce_model_v4 --split test \
  --pairs work/s4_test_ce_pairs.parquet --dest work/s4_test_ce.parquet --bs 1024 --max-len 128
aws s3 cp work/s4_test_ce.parquet s3://$B/v4/s4_test_ce.parquet --only-show-errors
echo "RUN_FINISHED"
