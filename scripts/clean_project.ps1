param(
    [switch]$IncludeDataCache
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot

$pathsToRemove = @(
    (Join-Path $projectDir 'build'),
    (Join-Path $projectDir 'dist'),
    (Join-Path $PSScriptRoot '__pycache__')
)

if ($IncludeDataCache) {
    $pathsToRemove += @(
        (Join-Path $projectDir 'data\MNIST\raw'),
        (Join-Path $projectDir 'test_images')
    )
}

foreach ($path in $pathsToRemove) {
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removed: $path"
    }
}

Write-Host 'Project cleanup completed.'
