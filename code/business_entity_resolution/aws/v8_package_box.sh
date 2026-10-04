#!/bin/bash
# usage: bash v8/package_box.sh <matching_dir> [candidate_dir]   (run in /opt/er on the CPU box)
set -euo pipefail
B=er-challenge-656791094360; M=$1; C=${2:-$1}
rm -rf /opt/er/pkg && mkdir -p /opt/er/pkg && cd /opt/er/pkg
aws s3 sync s3://$B/v8pkg/code code --only-show-errors
aws s3 cp s3://$B/v8pkg/Documentation_template.md Documentation_template.md --only-show-errors
python - "$M" "$C" <<'PY'
import sys, zipfile, os
m, c = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile("/opt/er/COMPUTATION_ISSUE_submission.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6)
z.write(f"/opt/er/{m}/matching_results.tsv", "output/matching_results.tsv")
z.write(f"/opt/er/{c}/candidate_pairs.tsv", "output/candidate_pairs.tsv")
for root, _, files in os.walk("code"):
    for f in files:
        if "__pycache__" not in root: z.write(os.path.join(root, f), os.path.join(root, f))
z.write("Documentation_template.md", "Documentation_template.md")
z.close()
print("zip ok", os.path.getsize("/opt/er/COMPUTATION_ISSUE_submission.zip") / 1e6, "MB")
PY
cd /opt/er && python code/utils/validate_submission.py --matching $M/matching_results.tsv --candidate $C/candidate_pairs.tsv --test-dir dataset/test | tail -3
aws s3 cp /opt/er/COMPUTATION_ISSUE_submission.zip s3://$B/v8pkg/COMPUTATION_ISSUE_submission.zip --only-show-errors
aws s3 cp $M/matching_results.tsv s3://$B/v8pkg/final_matching_results.tsv --only-show-errors
echo PACKAGE_DONE
