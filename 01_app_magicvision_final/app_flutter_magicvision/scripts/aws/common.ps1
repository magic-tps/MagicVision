Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'config.ps1')
function Invoke-MagicVisionStep([string]$Step) {
    # Only non-secret project settings cross this temporary JSON file.
    $temporaryConfig = [IO.Path]::GetTempFileName()
    $originalOutputEncoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false
        $json = $MagicVisionConfig | ConvertTo-Json -Depth 10
        [IO.File]::WriteAllText($temporaryConfig, $json, (New-Object Text.UTF8Encoding $false))
        & $MagicVisionConfig.python (Join-Path $PSScriptRoot 'workflow.py') --config $temporaryConfig --step $Step
        if ($LASTEXITCODE -ne 0) { throw "MagicVision step '$Step' stopped (exit $LASTEXITCODE)." }
    } finally {
        [Console]::OutputEncoding = $originalOutputEncoding
        Remove-Item -LiteralPath $temporaryConfig -Force
    }
}
