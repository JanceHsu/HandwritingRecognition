@echo off
setlocal

set "ROOT=%~dp0"
set "DIST=%ROOT%dist"
set "QT_BIN=D:\Develop\Qt\6.11.1\msvc2022_64\bin"

if "%LIBTORCH_DIR%"=="" (
    set "LIBTORCH_DIR=D:\Develop\libtorch"
)

set "LIBTORCH_BIN=%LIBTORCH_DIR%\lib"

if "%LIBTORCH_DEVICE%"=="" (
    set "LIBTORCH_DEVICE=auto"
)

if not exist "%DIST%\handwriting_recog.exe" (
    echo Release package not found. Building package first...
    powershell -ExecutionPolicy Bypass -File "%ROOT%scripts\package_release.ps1" -LibtorchDir "%LIBTORCH_DIR%"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$buildExe = Join-Path '%ROOT%build\Release' 'handwriting_recog.exe'; $distExe = Join-Path '%DIST%' 'handwriting_recog.exe'; if (!(Test-Path $distExe) -or ((Test-Path $buildExe) -and ((Get-Item $buildExe).LastWriteTimeUtc -gt (Get-Item $distExe).LastWriteTimeUtc))) { & '%ROOT%scripts\package_release.ps1' -LibtorchDir '%LIBTORCH_DIR%' }"

set "PATH=%QT_BIN%;%LIBTORCH_BIN%;%PATH%"
pushd "%DIST%"
start "" "handwriting_recog.exe"
popd
