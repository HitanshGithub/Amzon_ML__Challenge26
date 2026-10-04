#!/bin/bash
# EC2 user-data: pull code + data from S3, run the full pipeline, push results back, power off.
# The instance is launched with --instance-initiated-shutdown-behavior stop, so "poweroff" stops billing
# for compute while keeping the disk (and all intermediate files) for a quick re-run.
set -uxo pipefail
BUCKET="__BUCKET__"
STAGES="__STAGES__"
RUN_ID="__RUN_ID__"
HOME_DIR=/opt/er
LOG=$HOME_DIR/run.log
mkdir -p $HOME_DIR && cd $HOME_DIR
exec > >(tee -a $LOG) 2>&1

export DEBIAN_FRONTEND=noninteractive
if [ ! -d venv ]; then
  apt-get update -y && apt-get install -y python3-venv python3-pip
  python3 -m venv venv
fi
source venv/bin/activate
command -v aws >/dev/null || pip install -q awscli

push_log() { aws s3 cp $LOG "s3://$BUCKET/runs/$RUN_ID/run.log" --only-show-errors || true; }
( while true; do sleep 120; push_log; done ) &

aws s3 sync "s3://$BUCKET/code/business_entity_resolution" code --delete --only-show-errors
pip install --upgrade pip -q && pip install -q -r code/requirements.txt
[ -d dataset/train ] || aws s3 sync "s3://$BUCKET/dataset" dataset --only-show-errors
nproc; free -g; df -h /; nvidia-smi || true

python code/src/run_pipeline.py --data-dir dataset --work-dir work --out-dir output \
  --stages "$STAGES" --n-jobs $(( $(nproc) - 1 ))
STATUS=$?
echo "PIPELINE_EXIT=$STATUS"

if [ -f output/matching_results.tsv ]; then
  python code/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test || true
fi
aws s3 sync output "s3://$BUCKET/runs/$RUN_ID/output" --only-show-errors
for f in train_report.json model.txt translit.json; do
  [ -f work/$f ] && aws s3 cp work/$f "s3://$BUCKET/runs/$RUN_ID/$f" --only-show-errors
done
echo "RUN_FINISHED status=$STATUS"
push_log
poweroff
