#!/bin/bash
set -uxo pipefail
B=er-challenge-656791094360
export HOME=/root
H=/opt/er; LOG=$H/v8_boot.log
mkdir -p $H && cd $H
exec > >(tee -a $LOG) 2>&1
export DEBIAN_FRONTEND=noninteractive
if [ ! -d venv ]; then apt-get update -y && apt-get install -y python3-venv python3-pip; python3 -m venv venv; fi
source venv/bin/activate
command -v aws >/dev/null || pip install -q awscli
aws s3 sync "s3://$B/code/business_entity_resolution" code --delete --only-show-errors
pip install --upgrade pip -q && pip install -q -r code/requirements.txt
mkdir -p v8 work dataset/train dataset/test
aws s3 cp "s3://$B/v8/ce8.py" v8/ce8.py --only-show-errors
for f in test_source1 test_source2 test_source3; do aws s3 cp "s3://$B/dataset/test/$f.tsv" dataset/test/$f.tsv --only-show-errors & done
for f in train_source1 train_source2 train_source3 train_ground_truth; do aws s3 cp "s3://$B/dataset/train/$f.tsv" dataset/train/$f.tsv --only-show-errors & done
wait
echo "V8_BOOT_READY $(date -u +%H:%M:%S)"
aws s3 cp $LOG "s3://$B/v8/boot_$(hostname).log" --only-show-errors
