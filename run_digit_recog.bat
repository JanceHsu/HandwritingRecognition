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

if not exist "%DIST%\digit_recog.exe" (
    echo Release package not found. Building package first...
    powershell -ExecutionPolicy Bypass -File "%ROOT%scripts\package_release.ps1" -LibtorchDir "%LIBTORCH_DIR%"
)

set "PATH=%QT_BIN%;%LIBTORCH_BIN%;%PATH%"
pushd "%DIST%"
start "" "digit_recog.exe"
popd