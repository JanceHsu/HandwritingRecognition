param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$BuildDir,
    [string]$QtBin = "D:\Develop\Qt\6.11.1\msvc2022_64\bin",
    [string]$LibtorchDir = "D:\Develop\libtorch",
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'

if (-not $BuildDir) {
    $BuildDir = Join-Path $ProjectDir 'build'
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectDir 'dist'
}

if ($env:LIBTORCH_DIR -and -not $PSBoundParameters.ContainsKey('LibtorchDir')) {
    $LibtorchDir = $env:LIBTORCH_DIR
}

$releaseDir = Join-Path $BuildDir 'Release'
$exePath = Join-Path $releaseDir 'handwriting_recog.exe'

if (-not (Test-Path $exePath)) {
    throw "Executable not found: $exePath"
}

if (Test-Path $OutputDir) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Copy-Item $exePath $OutputDir -Force

$artifactDir = Join-Path $ProjectDir 'artifacts'
$modelSource = Join-Path $artifactDir 'models\mnist_model.pt'
if (-not (Test-Path $modelSource)) {
    throw "Model file not found: $modelSource"
}

$modelDir = Join-Path $OutputDir 'models'
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Copy-Item $modelSource (Join-Path $modelDir 'mnist_model.pt') -Force

$modelStateSource = Join-Path $artifactDir 'models\model.pth'
if (Test-Path $modelStateSource) {
    Copy-Item $modelStateSource (Join-Path $modelDir 'model.pth') -Force
}

$windeployqt = Join-Path $QtBin 'windeployqt.exe'
if (-not (Test-Path $windeployqt)) {
    throw "windeployqt not found: $windeployqt"
}

Push-Location $OutputDir
try {
    & $windeployqt --release --compiler-runtime .\handwriting_recog.exe
} finally {
    Pop-Location
}

$launcherPath = Join-Path $OutputDir 'run_handwriting_recog.bat'
@"
@echo off
setlocal
set "PATH=$QtBin;$LibtorchDir\lib;%PATH%"
if "%LIBTORCH_DEVICE%"=="" (
    set "LIBTORCH_DEVICE=auto"
)
start "" "%~dp0handwriting_recog.exe"
"@ | Set-Content -Path $launcherPath -Encoding ASCII

$libDir = Join-Path $LibtorchDir 'lib'
$dlls = Get-ChildItem $libDir -Filter *.dll -ErrorAction Stop
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName $OutputDir -Force
}

Write-Host "Packaged release to $OutputDir"