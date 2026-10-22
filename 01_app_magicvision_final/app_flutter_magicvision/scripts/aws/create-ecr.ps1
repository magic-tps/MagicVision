# Run from any working directory. See docs/aws-deployment.md.
. (Join-Path $PSScriptRoot 'common.ps1')
Invoke-MagicVisionStep 'create-ecr'

