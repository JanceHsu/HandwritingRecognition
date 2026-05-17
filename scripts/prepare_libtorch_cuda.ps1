param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string]$CudaLibtorchDir = "D:\Develop\libtorch-cuda",
    [bool]$StopRunningApp = $true
)

$ErrorActionPreference = 'Stop'

if ($StopRunningApp) {
    $procs = Get-Process -Name digit_recog -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force
        Write-Host "Stopped digit_recog process count:" $procs.Count
    } else {
        Write-Host "No running digit_recog process."
    }
}

$torchCmakeDir = Join-Path $CudaLibtorchDir 'share\cmake\Torch'
$torchLibDir = Join-Path $CudaLibtorchDir 'lib'

if (-not (Test-Path $CudaLibtorchDir)) {
    throw "CUDA LibTorch directory not found: $CudaLibtorchDir"
}
if (-not (Test-Path $torchCmakeDir)) {
    throw "Torch CMake config not found: $torchCmakeDir"
}
if (-not (Test-Path $torchLibDir)) {
    throw "Torch lib directory not found: $torchLibDir"
}

Write-Host "CUDA LibTorch precheck passed."
Write-Host "ProjectDir: $ProjectDir"
Write-Host "CudaLibtorchDir: $CudaLibtorchDir"
Write-Host ""
Write-Host "Next build command (MSVC):"
Write-Host ('cmake -S {0} -B {0}\build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="D:/Develop/Qt/6.11.1/msvc2022_64;{1}" -DTorch_DIR="{1}/share/cmake/Torch" -DDIGIT_RECOG_DEFAULT_DEVICE=auto' -f $ProjectDir, $CudaLibtorchDir)
Write-Host "cmake --build $ProjectDir\build --config Release"
Write-Host ""
Write-Host "Runtime tip: set LIBTORCH_DIR=$CudaLibtorchDir and LIBTORCH_DEVICE=cuda (or auto)."
