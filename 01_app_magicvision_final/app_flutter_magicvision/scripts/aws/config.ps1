# Configuration only. Credentials belong to AWS CLI/SSO, never to this file.
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
function Get-MagicVisionSetting([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}
$AWS_REGION = Get-MagicVisionSetting 'AWS_REGION' (Get-MagicVisionSetting 'AWS_DEFAULT_REGION' '')
$AWS_ACCOUNT_ID = '' # Always resolved and verified using STS, never guessed.
$S3_BUCKET = Get-MagicVisionSetting 'MV_S3_BUCKET' '' # magicvision-ACCOUNT-REGION
$ECR_REPOSITORY = Get-MagicVisionSetting 'MV_ECR_REPOSITORY' 'magicvision-yolo26m'
$MODEL_NAME = Get-MagicVisionSetting 'MV_MODEL_NAME' 'magicvision-yolo26m'
$ENDPOINT_CONFIG_NAME = Get-MagicVisionSetting 'MV_ENDPOINT_CONFIG_NAME' 'magicvision-yolo26m-async'
$ENDPOINT_NAME = Get-MagicVisionSetting 'MV_ENDPOINT_NAME' 'magicvision-yolo26m-async'
$MagicVisionConfig = @{
    root = $ProjectRoot
    region = $AWS_REGION
    bucket = $S3_BUCKET
    repository = $ECR_REPOSITORY
    model = $MODEL_NAME
    endpoint_config = $ENDPOINT_CONFIG_NAME
    endpoint = $ENDPOINT_NAME
    instance = Get-MagicVisionSetting 'MV_INSTANCE_TYPE' 'ml.g4dn.xlarge'
    role_arn = Get-MagicVisionSetting 'MV_EXECUTION_ROLE_ARN' ''
    torch_flavor = Get-MagicVisionSetting 'MV_TORCH_FLAVOR' 'cu118'
    local_device = Get-MagicVisionSetting 'MV_LOCAL_DEVICE' 'cpu'
    test_image = Get-MagicVisionSetting 'MV_TEST_IMAGE' ''
    python = Get-MagicVisionSetting 'MV_PYTHON' 'python'
    api_stack = Get-MagicVisionSetting 'MV_API_STACK_NAME' 'magicvision-async-api'
    api_role_arn = Get-MagicVisionSetting 'MV_API_EXECUTION_ROLE_ARN' ''
    allowed_client_cidr = Get-MagicVisionSetting 'MV_ALLOWED_CLIENT_CIDR' ''
}
