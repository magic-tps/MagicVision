param(
    [string]$Model = "lfm2-vl-1.6b",
    [string]$Package = "com.example.mi_app",
    [switch]$DownloadOnly
)

$ErrorActionPreference = "Stop"

$models = @{
    "lfm2-vl-450m" = @{
        Url = "https://vlqqczxwyaodtcdmdmlw.supabase.co/storage/v1/object/public/cactus-models/lfm2-vl-450m.zip"
        SizeMb = 420
    }
    "lfm2-vl-1.6b" = @{
        Url = "https://vlqqczxwyaodtcdmdmlw.supabase.co/storage/v1/object/public/cactus-models/lfm2-vl-1.6b.zip"
        SizeMb = 1440
    }
}

if (-not $models.ContainsKey($Model)) {
    throw "Modelo no reconocido: $Model. Disponibles: $($models.Keys -join ', ')"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$cacheDir = Join-Path $repoRoot ".mobilevlm-cache"
$zipPath = Join-Path $cacheDir "$Model.zip"
$extractDir = Join-Path $cacheDir $Model
$remoteDir = "/sdcard/Android/data/$Package/files/models/$Model"
$privateDir = "/data/user/0/$Package/app_flutter/models/$Model"

New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

if (-not (Test-Path -LiteralPath $zipPath)) {
    Write-Host "Descargando $Model ($($models[$Model].SizeMb) MB aprox.)..."
    Invoke-WebRequest -Uri $models[$Model].Url -OutFile $zipPath
} else {
    Write-Host "Zip local encontrado: $zipPath"
}

$hasExtractedFiles = (Test-Path -LiteralPath $extractDir) -and
    ((Get-ChildItem -LiteralPath $extractDir -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1) -ne $null)

if (-not $hasExtractedFiles) {
    Write-Host "Extrayendo modelo en $extractDir..."
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
} else {
    Write-Host "Modelo ya extraido: $extractDir"
}

$appleOnlyDirs = @(
    "audio_encoder.mlmodelc",
    "audio_encoder.mlpackage",
    "vision_encoder.mlmodelc",
    "vision_encoder.mlpackage"
)
foreach ($dir in $appleOnlyDirs) {
    $target = Join-Path $extractDir $dir
    if (Test-Path -LiteralPath $target) {
        Write-Host "Omitiendo carpeta Apple no necesaria en Android: $dir"
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

if ($DownloadOnly) {
    Write-Host "Listo. Modelo preparado en PC: $extractDir"
    exit 0
}

$adb = Get-Command adb -ErrorAction SilentlyContinue
if ($null -eq $adb) {
    $adbCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"),
        (Join-Path $env:USERPROFILE "AppData\Local\Android\Sdk\platform-tools\adb.exe")
    )
    $adbPath = $adbCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($null -eq $adbPath) {
        throw "No encuentro adb. Instala Android SDK Platform Tools o abre Android Studio y revisa SDK Manager."
    }
} else {
    $adbPath = $adb.Source
}

$devices = & $adbPath devices
$connected = $devices | Select-String -Pattern "device$" | Select-Object -First 1
if ($null -eq $connected) {
    throw "No hay telefono Android conectado por ADB. Activa Depuracion USB y vuelve a ejecutar."
}

Write-Host "Creando carpeta en el telefono: $remoteDir"
& $adbPath shell "mkdir -p '$remoteDir'"

Write-Host "Copiando modelo al telefono..."
& $adbPath push "$extractDir\." "$remoteDir"

Write-Host "Copiando modelo a almacenamiento privado de la app..."
$tempPrivateDir = "$privateDir.tmp"
$privateCopyOk = $false
& $adbPath shell run-as $Package rm -rf $tempPrivateDir
& $adbPath shell run-as $Package rm -rf $privateDir
& $adbPath shell run-as $Package mkdir -p $tempPrivateDir
& $adbPath shell run-as $Package cp -R "$remoteDir/." "$tempPrivateDir/"
if ($LASTEXITCODE -eq 0) {
    & $adbPath shell run-as $Package mv $tempPrivateDir $privateDir
}

if ($LASTEXITCODE -eq 0) {
    & $adbPath shell run-as $Package test -f "$privateDir/config.txt"
    $hasConfig = $LASTEXITCODE -eq 0
    & $adbPath shell run-as $Package test -f "$privateDir/tokenizer.json"
    $hasTokenizer = $LASTEXITCODE -eq 0
    $privateCopyOk = $hasConfig -and $hasTokenizer
}

if ($privateCopyOk) {
    Write-Host "Limpiando copia externa temporal..."
    & $adbPath shell "rm -rf '$remoteDir'"
} else {
    Write-Warning "No pude usar run-as para copiar a almacenamiento privado. Dejare la copia externa con permisos de lectura."
    & $adbPath shell "chmod -R u+rwX,g+rwX,o+rX '$remoteDir'"
}

Write-Host "Listo. Abre la app y di: estado modelo visual"
