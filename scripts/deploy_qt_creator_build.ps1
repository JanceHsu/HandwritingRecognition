param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$BuildRoot,
    [string]$TargetConfig = 'release',
    [string]$TargetName = 'handwriting_recog',
    [string]$QtBin = "D:\Develop\Qt\6.11.1\msvc2022_64\bin",
    [string]$LibtorchDir = "D:\Develop\libtorch"
)

$ErrorActionPreference = 'Stop'

if (-not $BuildRoot) {
    throw 'BuildRoot is required.'
}

$preferredExePath = Join-Path $BuildRoot "$TargetName.exe"
if (Test-Path $preferredExePath) {
    $exePath = $preferredExePath
} else {
    $preferredExePath = Join-Path (Join-Path $BuildRoot $TargetConfig) "$TargetName.exe"
    if (Test-Path $preferredExePath) {
        $exePath = $preferredExePath
    } else {
        $exeCandidates = Get-ChildItem -Path $BuildRoot -Recurse -Filter "$TargetName.exe" -File | Sort-Object FullName
        if (-not $exeCandidates) {
            throw "Executable not found under $BuildRoot for target $TargetName"
        }
        $exePath = $exeCandidates[0].FullName
    }
}

if (-not (Test-Path $exePath)) {
    throw "Executable not found: $exePath"
}

$modelSourceDir = Join-Path $ProjectDir 'artifacts\models'
if (Test-Path $modelSourceDir) {
    $destinationDir = Join-Path (Split-Path -Parent $exePath) 'models'
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    foreach ($fileName in @('mnist_model.pt', 'model.pth')) {
        $sourceFile = Join-Path $modelSourceDir $fileName
        if (Test-Path $sourceFile) {
            Copy-Item $sourceFile (Join-Path $destinationDir $fileName) -Force
        }
    }
} else {
    Write-Warning "Model directory not found, skipping: $modelSourceDir"
}

$libDir = Join-Path $LibtorchDir 'lib'
if (Test-Path $libDir) {
    Get-ChildItem $libDir -Filter *.dll | ForEach-Object {
        Copy-Item $_.FullName (Split-Path -Parent $exePath) -Force
    }
}

$windeployqt = Join-Path $QtBin 'windeployqt.exe'
if (Test-Path $windeployqt) {
    $targetLeaf = Split-Path -Leaf $exePath
    $windeployqtFlag = if ($TargetConfig -eq 'debug') { '--debug' } else { '--release' }
    Push-Location (Split-Path -Parent $exePath)
    try {
        & $windeployqt $windeployqtFlag --compiler-runtime .\$targetLeaf
    } finally {
        Pop-Location
    }
}

Write-Host "Qt Creator build deployed to $(Split-Path -Parent $exePath)"