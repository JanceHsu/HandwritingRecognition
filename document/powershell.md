# PowerShell 脚本说明

本项目在 `scripts/` 目录下维护了若干 PowerShell 脚本，用于构建部署、发布打包、环境清理等工作。以下逐一说明每个脚本的用途、参数和执行逻辑。

---

## 1. deploy_qt_creator_build.ps1

**路径：** `scripts/deploy_qt_creator_build.ps1`

**用途：** Qt Creator 构建的自动部署脚本。在 Qt Creator 完成编译后，由 `.pro` 文件中的 `QMAKE_POST_LINK` 自动调用，负责将模型文件、LibTorch DLL 和 Qt 运行库复制到可执行文件所在目录，使程序可以直接运行。

**调用方式：** 由 Qt Creator 构建系统自动调用，无需手动执行。`.pro` 文件中的调用命令为：

```qmake
QMAKE_POST_LINK += powershell -NoProfile -ExecutionPolicy Bypass -File $$PWD/scripts/deploy_qt_creator_build.ps1 -ProjectDir $$PWD -BuildRoot $$OUT_PWD -TargetConfig release -TargetName $$TARGET -LibtorchDir $$LIBTORCH_DIR
```

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ProjectDir` | 脚本上级目录 | 项目根目录，用于定位 `artifacts/models` |
| `BuildRoot` | （必填） | Qt Creator 的构建输出根目录（即 `$$OUT_PWD`） |
| `TargetConfig` | `release` | 构建配置，用于定位可执行文件（如 `release/handwriting_recog.exe`） |
| `TargetName` | `handwriting_recog` | 可执行文件名（不含 `.exe`） |
| `QtBin` | `D:\Develop\Qt\6.11.1\msvc2022_64\bin` | Qt 的 `bin` 目录，用于调用 `windeployqt` |
| `LibtorchDir` | `D:\Develop\libtorch` | LibTorch 安装目录，用于复制 DLL |

**执行流程：**

1. 在 `BuildRoot` 下查找可执行文件（优先 `$BuildRoot/$TargetConfig/$TargetName.exe`，找不到则递归搜索）
2. 将 `artifacts/models/` 下的 `mnist_model.pt` 和 `model.pth` 复制到可执行文件同级的 `models/` 目录
3. 将 `LibtorchDir/lib/` 下的所有 DLL 复制到可执行文件同级目录
4. 调用 `windeployqt` 部署 Qt 运行库（根据 `TargetConfig` 选择 `--debug` 或 `--release`）

---

## 2. package_release.ps1

**路径：** `scripts/project_operation/package_release.ps1`

**用途：** 将 CMake Release 构建的产物打包为可分发的发布目录。收集可执行文件、模型文件、Qt 运行库、LibTorch DLL，并生成一键启动脚本 `run_handwriting_recog.bat`。

**调用方式：**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\project_operation\package_release.ps1
```

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ProjectDir` | 脚本上级目录 | 项目根目录 |
| `BuildDir` | `$ProjectDir\build` | CMake 构建目录 |
| `QtBin` | `D:\Develop\Qt\6.11.1\msvc2022_64\bin` | Qt 的 `bin` 目录 |
| `LibtorchDir` | `D:\Develop\libtorch`（或 `$env:LIBTORCH_DIR`） | LibTorch 安装目录 |
| `OutputDir` | `$ProjectDir\dist` | 发布输出目录 |

**执行流程：**

1. 从 `build/Release/` 复制 `handwriting_recog.exe` 到 `dist/`
2. 将 `artifacts/models/` 下的模型文件复制到 `dist/models/`
3. 调用 `windeployqt --release` 部署 Qt 运行库到 `dist/`
4. 生成 `dist/run_handwriting_recog.bat` 启动脚本（自动设置 PATH 包含 Qt 和 LibTorch 路径）
5. 将 `LibtorchDir/lib/` 下的所有 DLL 复制到 `dist/`

**输出产物：** `dist/` 目录，包含完整的可分发程序，可直接双击 `run_handwriting_recog.bat` 启动。

---

## 3. clean_project.ps1

**路径：** `scripts/project_operation/clean_project.ps1`

**用途：** 清理项目构建产物和临时文件，将项目恢复到干净状态。

**调用方式：**

```powershell
# 基本清理（构建目录 + 发布目录 + Python 缓存）
powershell -ExecutionPolicy Bypass -File scripts\project_operation\clean_project.ps1

# 包含 MNIST 数据缓存和测试图片
powershell -ExecutionPolicy Bypass -File scripts\project_operation\clean_project.ps1 -IncludeDataCache
```

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `IncludeDataCache` | `$false` | 开关参数，是否同时清理 MNIST 数据集缓存和测试图片 |

**清理范围：**

| 路径 | 始终清理 | 说明 |
|------|----------|------|
| `build/` | 是 | CMake 和 Qt Creator 的所有构建产物 |
| `dist/` | 是 | 打包生成的发布目录 |
| `scripts/__pycache__/` | 是 | Python 字节码缓存 |
| `data/MNIST/raw/` | 仅 `-IncludeDataCache` | 下载的 MNIST 数据集原始文件 |
| `test_images/` | 仅 `-IncludeDataCache` | 测试用图片 |

---

## 4. prepare_libtorch_cuda.ps1

**路径：** `scripts/project_operation/prepare_libtorch_cuda.ps1`

**用途：** CUDA 版 LibTorch 的环境预检脚本。验证指定的 CUDA LibTorch 目录结构是否完整，并输出后续的 CMake 构建命令。不会修改任何文件，仅做检查和提示。

**调用方式：**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\project_operation\prepare_libtorch_cuda.ps1 -CudaLibtorchDir D:\Develop\libtorch-cuda
```

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ProjectDir` | 脚本上级目录 | 项目根目录 |
| `CudaLibtorchDir` | `D:\Develop\libtorch-cuda` | CUDA 版 LibTorch 的安装目录 |
| `StopRunningApp` | `$true` | 是否先终止正在运行的 `handwriting_recog` 进程 |

**执行流程：**

1. 如果 `StopRunningApp` 为 `$true`，查找并终止所有 `handwriting_recog` 进程
2. 检查 `CudaLibtorchDir` 目录是否存在
3. 检查 `share/cmake/Torch` 目录是否存在（CMake 配置文件）
4. 检查 `lib/` 目录是否存在（DLL 和导入库）
5. 输出下一步的 CMake 构建命令（可直接复制执行）

**注意：** 此脚本仅做环境检查和命令提示，不会自动执行构建。构建命令中包含已弃用的 `HANDWRITING_RECOG_DEFAULT_DEVICE` 参数，实际使用时可忽略该参数。

---

## 脚本调用关系

```
Qt Creator 构建
  └─ .pro (QMAKE_POST_LINK)
       └─ deploy_qt_creator_build.ps1    ← 自动调用

CMake 构建 + 打包
  └─ cmake --build build --config Release
  └─ project_operation/package_release.ps1  ← 手动调用

环境准备
  └─ project_operation/prepare_libtorch_cuda.ps1  ← 手动调用（仅检查）

项目清理
  └─ project_operation/clean_project.ps1  ← 手动调用
```
