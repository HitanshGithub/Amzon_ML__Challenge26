# Launch one g6.8xlarge that runs the whole pipeline and stops itself when finished.
# Usage (from code/business_entity_resolution):  powershell -File aws\launch.ps1 [-Spot] [-Stages "..."]
param(
    [string]$Bucket = "er-challenge-656791094360",
    [string]$InstanceType = "g6.8xlarge",
    [string]$Region = "us-east-1",
    [string]$Stages = "translit,prepare,block,train,predict",
    [string]$Bootstrap = "bootstrap.sh",
    [int]$DiskGB = 200,
    [string]$SubnetId = "",
    [switch]$Spot
)
$ErrorActionPreference = "Stop"
$env:AWS_DEFAULT_REGION = $Region
$here = Split-Path -Parent $PSScriptRoot           # code/business_entity_resolution
$role = "er-challenge-ec2"
$runId = Get-Date -Format "yyyyMMdd-HHmmss"

# 1. IAM role: read/write only this bucket + SSM (remote commands without SSH / open ports)
$existing = aws iam list-roles --query "Roles[?RoleName=='$role'].RoleName" --output text
if (-not $existing) {
    $trust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    $trustFile = New-TemporaryFile; Set-Content $trustFile $trust -Encoding ascii
    aws iam create-role --role-name $role --assume-role-policy-document "file://$trustFile" | Out-Null
    $pol = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:ListBucket","s3:DeleteObject"],"Resource":["arn:aws:s3:::' + $Bucket + '","arn:aws:s3:::' + $Bucket + '/*"]}]}'
    $polFile = New-TemporaryFile; Set-Content $polFile $pol -Encoding ascii
    aws iam put-role-policy --role-name $role --policy-name s3-bucket --policy-document "file://$polFile"
    aws iam attach-role-policy --role-name $role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    aws iam create-instance-profile --instance-profile-name $role | Out-Null
    aws iam add-role-to-instance-profile --instance-profile-name $role --role-name $role
    Start-Sleep 15   # IAM propagation
}

# 2. Upload code
aws s3 sync $here "s3://$Bucket/code/business_entity_resolution" --delete --exclude "*__pycache__*" --only-show-errors

# 3. User data
$ud = (Get-Content (Join-Path $PSScriptRoot $Bootstrap) -Raw).Replace("__BUCKET__", $Bucket).Replace("__STAGES__", $Stages).Replace("__RUN_ID__", $runId).Replace("`r`n", "`n")
$udFile = New-TemporaryFile; [IO.File]::WriteAllText($udFile, $ud)

# 4. AMI: Ubuntu 24.04 (Python 3.12). GPU types get the Deep Learning Base image with NVIDIA drivers.
if ($InstanceType -match '^(g|p)\d') {
    $amiParam = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
} else {
    $amiParam = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}
$ami = aws ssm get-parameter --name $amiParam --query Parameter.Value --output text

$cli = @("ec2", "run-instances", "--image-id", $ami, "--instance-type", $InstanceType,
    "--iam-instance-profile", "Name=$role",
    "--block-device-mappings", "DeviceName=/dev/sda1,Ebs={VolumeSize=$DiskGB,VolumeType=gp3,DeleteOnTermination=true}",
    "--instance-initiated-shutdown-behavior", "stop",
    "--user-data", "file://$udFile",
    "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=er-challenge},{Key=run,Value=$runId}]",
    "--query", "Instances[0].InstanceId", "--output", "text")
if ($SubnetId) { $cli += @("--subnet-id", $SubnetId) }
if ($Spot) { $cli += @("--instance-market-options", "MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}") }
$id = aws @cli
Write-Output "instance=$id run=$runId"
Write-Output "log: aws s3 cp s3://$Bucket/runs/$runId/run.log -"
