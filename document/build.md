# 构建与运行

本文档覆盖项目的全部构建、打包、运行和调试相关内容，包括命令行 CMake 和 Qt Creator qmake 两种构建方式的完整流程及其差异。

---

## 1. 环境准备

### 1.1 必要组件

- Windows 10/11 x64
- Visual Studio 2022 Build Tools（或完整 VS2022）
- Qt 6.11.1 MSVC 2022 x64
- CMake 3.30 或兼容版本
- CUDA Toolkit 12.8
- 与 CUDA 匹配的 LibTorch
- Python 3.12（隔空书写功能需要 `mediapipe` 和 `opencv-python`）

### 1.2 推荐目录

- Qt: `D:\Develop\Qt\6.11.1\msvc2022_64`
- LibTorch: `D:\Develop\libtorch`
- 项目: `D:\Develop\Project\Qt\HandwritingRecognition`

### 1.3 环境检查

```powershell
py -3.12 --version
& "D:\Develop\Qt\Tools\CMake_64\bin\cmake.exe" --version
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" --version
```

---

## 2. 命令行 CMake 构建

适用于可复现的发布流程、自动化构建和打包。

### 2.1 配置

先初始化 MSVC 环境：

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

然后运行 CMake 配置：

```bat
cmake -S D:\Develop\Project\Qt\HandwritingRecognition -B D:\Develop\Project\Qt\HandwritingRecognition\build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="D:/Develop/Qt/6.11.1/msvc2022_64;D:/Develop/libtorch" -DTorch_DIR="D:/Develop/libtorch/share/cmake/Torch"
```

`find_package(Torch)` 会自动检测 CUDA 支持，配置正确的编译定义和链接器选项。

### 2.2 构建

```bat
cmake --build D:\Develop\Project\Qt\HandwritingRecognition\build --config Release
```

编译成功后，可执行文件位于：`build\Release\handwriting_recog.exe`

### 2.3 CMakeLists.txt 说明

构建配置位于项目根目录的 `CMakeLists.txt`：

- `find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia)` — 查找 Qt6
- `find_package(Torch REQUIRED)` — 查找 LibTorch，自动配置 CUDA
- 源文件：`main.cpp`、`mainwindow.cpp`、`canvas.cpp`、`recognizer.cpp`、`airwritecontroller.cpp`
- Windows 额外链接：`mf`、`mfplat`、`mfreadwrite`、`mfuuid`、`ole32`、`shlwapi`（摄像头相关）
- 构建后自动复制 `artifacts/models` 到输出目录

---

## 3. Qt Creator qmake 构建

适用于日常开发调试，可以在 IDE 中直接断点、查看变量和 UI 状态。

### 3.1 导入项目

1. 在 Qt Creator 中打开 `HandwritingRecognition.pro`
2. 选择 Qt 6.11.1 / MSVC 2022 Kit
3. 点击构建

### 3.2 HandwritingRecognition.pro 说明

qmake 构建配置位于 `HandwritingRecognition.pro`：

- `LIBTORCH_DIR` 默认为 `D:/Develop/libtorch`，可通过 `LIBTORCH_DIR` 环境变量覆盖
- 手动设置 `INCLUDEPATH` 指向 LibTorch 头文件目录
- 通过 `exists($$LIBTORCH_DIR/lib/torch_cuda.lib)` 自动检测 CUDA 版 LibTorch 并链接相应库
- `QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ` — 强制包含 CUDA 初始化符号（见 3.3）
- `QMAKE_POST_LINK` 自动调用 `scripts/deploy_qt_creator_build.ps1` 部署 DLL 和模型文件

### 3.3 CUDA 链接器标志

CMake 的 `find_package(Torch)` 会自动向 MSVC 链接器注入 `-INCLUDE:?warp_size@cuda@at@@YAHXZ` 选项，强制包含 LibTorch CUDA 后端的初始化符号。qmake 构建缺少该选项时，链接器的死代码消除会连带移除 CUDA 初始化入口，导致运行时 `torch::cuda::is_available()` 返回 false。

`HandwritingRecognition.pro` 已在 CUDA 链接块中手动添加了该选项：

```qmake
exists($$LIBTORCH_DIR/lib/torch_cuda.lib) {
    LIBS += -lc10_cuda -ltorch_cuda -lcaffe2_nvrtc -lkineto
    QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ
    # ...
}
```

### 3.4 自动部署

构建完成后，`QMAKE_POST_LINK` 自动调用 `scripts/deploy_qt_creator_build.ps1`，该脚本执行：

1. 在构建目录中定位 `handwriting_recog.exe`
2. 将 `artifacts/models/` 下的模型文件复制到 exe 同级的 `models/` 目录
3. 将 `LibtorchDir/lib/` 下的所有 DLL 复制到 exe 同级目录
4. 调用 `windeployqt` 部署 Qt 运行库

---

## 4. 两种构建方式的差异

| 维度 | 命令行 CMake | Qt Creator qmake |
|------|-------------|------------------|
| 入口文件 | `CMakeLists.txt` | `HandwritingRecognition.pro` |
| 生成器 | Visual Studio 17 2022 | Qt Creator 内置 qmake |
| 适用场景 | 可复现发布、打包、自动化 | IDE 调试、UI 迭代、断点调试 |
| CUDA 检测 | `find_package(Torch)` 自动配置 | `exists()` 条件判断 + 手动链接 |
| 链接器标志 | CMake 自动注入 `-INCLUDE:?warp_size@cuda@at@@YAHXZ` | 需在 `.pro` 中手动添加 |
| 构建目录 | `build/Release/` | `Desktop_Qt_6_11_1_MSVC2022_64bit-Release/` |
| 运行目录 | `build/Release` 或 `dist/` | Qt Creator 构建目录 |
| 依赖部署 | `package_release.ps1`（手动调用） | `deploy_qt_creator_build.ps1`（自动调用） |
| LibTorch 头文件 | CMake 通过 `Torch_DIR` 自动配置 | `.pro` 中手动设置 `INCLUDEPATH` |

两者编译的是同一套 `src/` 源码，差异主要在构建系统如何设置 include/lib 路径和运行时部署。运行时表现不一致时，优先检查运行目录里是否真的有 `torch_cuda.dll`、`c10_cuda.dll` 和 `models/`。

---

## 5. 清理旧产物

```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean_project.ps1
```

清理范围：`build/`、`dist/`、Python 缓存。加 `-IncludeDataCache` 参数还会清理 MNIST 数据缓存和测试图片。

---

## 6. 训练模型

```powershell
# GPU 正式训练
py -3.12 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models\gpu --device cuda --pin-memory --num-workers 8

# CPU 快速验证
py -3.12 scripts\train_mnist.py --epochs 3 --batch-size 256 --output-dir artifacts\models\cpu --device cpu --no-augment --num-workers 0
```

训练产物位于 `artifacts/models/{cpu,gpu}/`，包含 `mnist_model.pt`（TorchScript）和 `model.pth`（PyTorch state_dict）。详细训练技术细节参见 [model.md](model.md)。

---

## 7. 打包发布

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_release.ps1 -LibtorchDir D:\Develop\libtorch
```

打包脚本执行以下步骤：

1. 从 `build/Release/` 复制 `handwriting_recog.exe` 到 `dist/`
2. 将 `artifacts/models/` 下的模型文件按 CPU/GPU 分类复制到 `dist/models/`
3. 调用 `windeployqt --release` 部署 Qt 运行库到 `dist/`
4. 将 `LibtorchDir/lib/` 下的所有 DLL 复制到 `dist/`
5. 生成 `dist/run_handwriting_recog.bat` 启动脚本

发布包位于 `dist/` 目录，包含：`handwriting_recog.exe`、Qt 运行库、LibTorch 运行库（CUDA 版含 `torch_cuda.dll`、`c10_cuda.dll`）、`models/` 目录和一键启动脚本。

`build/` 是开发构建目录，`dist/` 是发布目录。只重新构建而不执行打包时，`dist/` 会保持旧快照。

---

## 8. 启动程序

### 8.1 一键启动（推荐）

```
run_handwriting_recog.bat
```

该脚本会检查 `dist/` 是否有最新构建，必要时自动调用 `package_release.ps1`，然后设置 PATH 并启动程序。

### 8.2 启动后检查

- 主窗口正常打开
- 左侧画板可写字
- CUDA 可用时复选框可勾选
- 点击"识别"不闪退
- 右侧操作日志显示模型加载状态

---

## 9. 隔空书写运行要求

- 系统 Python 3.12，安装 `mediapipe` 和 `opencv-python`
- 不创建新的虚拟环境
- 首次启动隔空书写时自动下载 MediaPipe hand landmarker 模型

---

## 10. 推荐工作流

1. 修改代码
2. 在 Qt Creator 中构建调试（qmake）
3. 确认功能正常后，命令行 cmake 构建 Release
4. 运行 `package_release.ps1` 打包
5. 双击 `run_handwriting_recog.bat` 验证

---

## 11. 常见问题

### 11.1 启动后提示入口点错误

加载到了错误版本的 Qt DLL。必须优先使用 `msvc2022_64\bin` 下的 Qt 运行库。

### 11.2 Qt Creator 构建后 CUDA 不可用

检查三项：构建目录里是否有 `torch_cuda.dll` 和 `c10_cuda.dll`；是否已执行 `deploy_qt_creator_build.ps1`；Kit 是否和 LibTorch 是同一套 MSVC 架构。

### 11.3 GPU 选项不显示

检查 CUDA 是否可用、LibTorch 是否是 CUDA 版本。

### 11.4 点击识别异常

确认模型文件存在、画板有数字、日志里推理设备符合预期。

### 11.5 LibTorch 和 Qt 的宏冲突

两者都定义了 `slots` 和 `signals` 宏。`src/recognizer.h` 通过在包含 LibTorch 头文件前先 `#undef` 这两个宏、包含后再恢复的方式解决。

---

## 12. 脚本一览

| 脚本 | 用途 |
|------|------|
| `scripts/train_mnist.py` | MNIST 模型训练与 TorchScript 导出 |
| `scripts/hand_tracker_service.py` | 隔空书写手部追踪服务 |
| `scripts/package_release.ps1` | CMake Release 构建打包 |
| `scripts/deploy_qt_creator_build.ps1` | Qt Creator 构建自动部署（QMAKE_POST_LINK 调用） |
| `scripts/clean_project.ps1` | 清理构建产物 |
| `scripts/prepare_libtorch_cuda.ps1` | CUDA 版 LibTorch 环境预检 |
| `scripts/export_test_images.py` | 导出 MNIST 测试图片 |
