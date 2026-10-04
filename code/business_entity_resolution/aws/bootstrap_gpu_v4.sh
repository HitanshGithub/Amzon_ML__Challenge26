#!/bin/bash
set -uxo pipefail
BUCKET="__BUCKET__"; RUN_ID="__RUN_ID__"
HOME_DIR=/opt/er; LOG=$HOME_DIR/v4_gpu.log
mkdir -p $HOME_DIR && cd $HOME_DIR
exec > >(tee -a $LOG) 2>&1
export DEBIAN_FRONTEND=noninteractive
if [ ! -d venv ]; then apt-get update -y && apt-get install -y python3-venv python3-pip; python3 -m venv venv; fi
source venv/bin/activate
command -v aws >/dev/null || pip install -q awscli
( while true; do sleep 120; aws s3 cp $LOG "s3://$BUCKET/runs/v4-20260926/v4_gpu.log" --only-show-errors || true; done ) &
aws s3 sync "s3://$BUCKET/code/business_entity_resolution" code --delete --only-show-errors
pip install --upgrade pip -q && pip install -q -r code/requirements.txt
bash code/aws/v4_gpu_body.sh
aws s3 cp $LOG "s3://$BUCKET/runs/v4-20260926/v4_gpu.log" --only-show-errors
poweroff
