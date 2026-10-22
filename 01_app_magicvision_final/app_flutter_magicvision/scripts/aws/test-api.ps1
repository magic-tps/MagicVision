Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'config.ps1')
$outputsPath = Join-Path $ProjectRoot 'backend\artifacts\aws\api-outputs.json'
$outputs = Get-Content -LiteralPath $outputsPath -Raw | ConvertFrom-Json
$apiUrl = [Uri]$outputs.ApiBaseUrl
if ($apiUrl.Scheme -ne 'https' -or $apiUrl.UserInfo -or $apiUrl.Query -or $apiUrl.Fragment) {
    throw 'The deployed API output must contain a plain HTTPS base URL.'
}
$imagePath = Join-Path $ProjectRoot 'ml\datasets\dataset_final_magicvision\images\test\coco_filtered_000000000872.jpg'
$reportPath = Join-Path $ProjectRoot 'backend\artifacts\aws\flutter-cloud-test.json'
Push-Location $ProjectRoot
try {
    # Test client uses system trusted CAs. No AWS credentials are passed to Flutter.
    & flutter test test/local_model_integration_test.dart --no-pub --reporter expanded `
        "--dart-define=LOCAL_INTEGRATION_URL=$($outputs.ApiBaseUrl)" `
        "--dart-define=LOCAL_INTEGRATION_IMAGE=$imagePath" `
        "--dart-define=LOCAL_INTEGRATION_REPORT=$reportPath" `
        '--dart-define=INTEGRATION_SCOPE=Deployed AWS API and real model/Dart; dataset camera; audible TTS and USB not tested'
    if ($LASTEXITCODE -ne 0) { throw 'Cloud API integration failed. Inspect the test output; older reports are not proof of this run.' }
} finally {
    Pop-Location
}
