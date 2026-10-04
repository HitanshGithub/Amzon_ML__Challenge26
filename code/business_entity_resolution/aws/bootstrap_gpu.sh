#!/bin/bash
# EC2 user-data for the GPU box (v3 cross-encoder): pull code + training data + v1 candidates,
# fine-tune the cross-encoder, push model + validation scores to S3, power off (instance stops).
set -uxo pipefail
BUCKET="__BUCKET__"
RUN_ID="__RUN_ID__"
HOME_DIR=/opt/er
LOG=$HOME_DIR/ce_train.log
mkdir -p $HOME_DIR && cd $HOME_DIR
exec > >(tee -a $LOG) 2>&1

export DEBIAN_FRONTEND=noninteractive
if [ ! -d venv ]; then
  apt-get update -y && apt-get install -y python3-venv python3-pip
  python3 -m venv venv
fi
source venv/bin/activate
command -v aws >/dev/null || pip install -q awscli
push_log() { aws s3 cp $LOG "s3://$BUCKET/runs/$RUN_ID/ce_train.log" --only-show-errors || true; }
( while true; do sleep 120; push_log; done ) &

aws s3 sync "s3://$BUCKET/code/business_entity_resolution" code --delete --only-show-errors
pip install --upgrade pip -q && pip install -q -r code/requirements.txt
mkdir -p dataset/train work
aws s3 sync "s3://$BUCKET/dataset/train" dataset/train --only-show-errors
for f in train_s1.parquet train_pool.parquet train_cands.parquet; do
  [ -f work/$f ] || aws s3 cp "s3://$BUCKET/v1work/$f" work/$f --only-show-errors
done
nvidia-smi; nproc; free -g
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -W ignore code/src/cross_encoder.py train --data-dir dataset --work-dir work \
  --cands work/train_cands.parquet --model intfloat/multilingual-e5-base --out work/ce_model \
  --n-s1 300000 --bs 256 --max-len 96 --workers 6
STATUS=$?
echo "CE_TRAIN_EXIT=$STATUS"
aws s3 sync work/ce_model "s3://$BUCKET/runs/$RUN_ID/ce_model" --only-show-errors
echo "RUN_FINISHED status=$STATUS"
push_log
poweroff
