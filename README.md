# 基于深度学习的手写数字识别系统

基于 Qt 6 + LibTorch 的桌面手写数字识别应用。用户可以在画板上用鼠标书写数字，也可以通过摄像头隔空书写，程序实时识别并显示结果。

## 功能

- **画板手写识别**：在 280x280 的白色画布上用鼠标书写 0-9 的数字，点击"识别"即可获得预测结果和可信度。
- **隔空书写**：开启摄像头后，伸出食指在空中比划，程序通过 MediaPipe 手部追踪捕捉指尖运动并映射到画布，实现非接触式书写。
- **推理设备切换**：支持 CPU 和 CUDA 两种推理设备，界面复选框一键切换，CUDA 模式自动预热。
- **操作日志**：右侧面板实时显示模型加载、识别结果、摄像头状态等关键事件。

## 技术栈

| 模块 | 技术 |
|------|------|
| 模型训练 | Python 3.12, PyTorch, Torchvision, CUDA 12.8 |
| 模型推理 | LibTorch (TorchScript), C++ 17 |
| 图形界面 | Qt 6 Widgets |
| 手部追踪 | OpenCV, MediaPipe Tasks, QProcess |
| 构建系统 | CMake 3.30, MSVC 2022, qmake |

## 项目结构

```
HandwritingRecognition/
├── src/                        # C++ 源码
│   ├── main.cpp                # 程序入口
│   ├── mainwindow.h/cpp        # 主窗口（UI 布局、设备切换、隔空书写调度）
│   ├── canvas.h/cpp            # 画布控件（鼠标绘制、程序化绘制）
│   ├── recognizer.h/cpp        # 识别器（LibTorch 模型加载、预处理、推理）
│   └── airwritecontroller.h/cpp # 隔空书写控制器（Python 进程管理）
├── scripts/                    # 脚本
│   ├── train_mnist.py          # MNIST 模型训练与 TorchScript 导出
│   ├── hand_tracker_service.py # 手部追踪服务（MediaPipe HandLandmarker）
│   ├── export_test_images.py   # 导出 MNIST 测试图片
│   ├── package_release.ps1     # 发布打包脚本
│   ├── deploy_qt_creator_build.ps1 # Qt Creator 构建部署脚本
│   ├── clean_project.ps1       # 项目清理脚本
│   └── prepare_libtorch_cuda.ps1   # CUDA 环境预检脚本
├── document/                   # 项目文档
├── artifacts/                  # 训练产物（models/cpu, models/gpu）
├── dist/                       # 发布包输出
├── CMakeLists.txt              # CMake 构建脚本
├── HandwritingRecognition.pro  # qmake 构建脚本（Qt Creator）
└── run_handwriting_recog.bat   # 一键启动脚本
```

## 环境要求

- Windows 10/11 x64
- Visual Studio 2022 Build Tools（或完整 VS2022）
- Qt 6.11.1 MSVC 2022 x64
- CUDA Toolkit 12.8
- LibTorch（与 CUDA 版本匹配）
- Python 3.12（隔空书写功能需要 `mediapipe` 和 `opencv-python`）

推荐安装路径：

```
D:\Develop\Qt\6.11.1\msvc2022_64
D:\Develop\libtorch
```

## 快速开始

### 1. 训练模型

```powershell
py -3.12 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models\gpu --device cuda --pin-memory --num-workers 8
```

训练完成后，`artifacts/models/gpu/` 下会生成 `mnist_model.pt` 和 `model.pth`。

### 2. 构建程序

**命令行 CMake（推荐用于发布）：**

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="D:/Develop/Qt/6.11.1/msvc2022_64;D:/Develop/libtorch" -DTorch_DIR="D:/Develop/libtorch/share/cmake/Torch"
cmake --build build --config Release
```

**Qt Creator（推荐用于调试）：**

打开 `HandwritingRecognition.pro`，选择 Qt 6.11.1 / MSVC 2022 Kit，点击构建即可。

### 3. 打包发布

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_release.ps1 -LibtorchDir D:\Develop\libtorch
```

### 4. 启动程序

```
run_handwriting_recog.bat
```

## 文档

| 文档 | 内容 |
|------|------|
| [构建与运行](document/build.md) | CMake 构建、Qt Creator 构建、差异对比、打包发布、常见问题 |
| [开发设计指导书](document/plan.md) | 分阶段开发全流程（环境准备→训练→推理→界面→集成→隔空书写→发布） |
| [模型训练、推理与设备管理](document/model.md) | CNN 结构、训练策略、预处理、CPU/CUDA 切换、预热 |
| [项目报告](document/report.md) | 背景、目标、方案、详细实现、实验结果、问题与解决 |
| [Python 脚本说明](document/python.md) | train_mnist.py、hand_tracker_service.py、export_test_images.py |
| [隔空书写架构说明](document/airwriting.md) | Qt-Python 通信、手势判定、食指锁定、稳定策略 |
| [PowerShell 脚本说明](document/powershell.md) | 打包、部署、清理、环境预检脚本 |
| [文件清单](document/files.md) | 项目全部文件与目录说明 |
| [环境快照](document/environment.txt) | Python、Qt、CUDA、LibTorch 版本记录 |
