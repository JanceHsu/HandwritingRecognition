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
$exePath = Join-Path $releaseDir 'digit_recog.exe'

if (-not (Test-Path $exePath)) {
    throw "Executable not found: $exePath"
}

if (Test-Path $OutputDir) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Copy-Item $exePath $OutputDir -Force

$artifactDir = Join-Path $ProjectDir 'artifacts'
$modelTargets = @(
    @{ Kind = 'cpu'; Source = @(
        (Join-Path $artifactDir 'models\cpu\mnist_model.pt')
    ) },
    @{ Kind = 'gpu'; Source = @(
        (Join-Path $artifactDir 'models\gpu\mnist_model.pt')
    ) }
)

foreach ($modelTarget in $modelTargets) {
    $modelDir = Join-Path (Join-Path $OutputDir 'models') $modelTarget.Kind
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    $copied = $false
    foreach ($candidate in $modelTarget.Source) {
        if (Test-Path $candidate) {
            Copy-Item $candidate (Join-Path $modelDir 'mnist_model.pt') -Force
            $candidateState = Join-Path (Split-Path $candidate -Parent) 'model.pth'
            if (Test-Path $candidateState) {
                Copy-Item $candidateState (Join-Path $modelDir 'model.pth') -Force
            }
            $copied = $true
            break
        }
    }

    if (-not $copied) {
        throw "Model file not found for $($modelTarget.Kind) target"
    }
}

$windeployqt = Join-Path $QtBin 'windeployqt.exe'
if (-not (Test-Path $windeployqt)) {
    throw "windeployqt not found: $windeployqt"
}

Push-Location $OutputDir
try {
    & $windeployqt --release --compiler-runtime .\digit_recog.exe
} finally {
    Pop-Location
}

$launcherPath = Join-Path $OutputDir 'run_digit_recog.bat'
@"
@echo off
setlocal
set "PATH=$QtBin;$LibtorchDir\lib;%PATH%"
if "%LIBTORCH_DEVICE%"=="" (
    set "LIBTORCH_DEVICE=auto"
)
start "" "%~dp0digit_recog.exe"
"@ | Set-Content -Path $launcherPath -Encoding ASCII

$libDir = Join-Path $LibtorchDir 'lib'
$dlls = Get-ChildItem $libDir -Filter *.dll -ErrorAction Stop
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName $OutputDir -Force
}

Write-Host "Packaged release to $OutputDir"