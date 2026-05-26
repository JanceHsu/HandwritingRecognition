# 项目构建全流程

本文档按“环境准备 -> 训练模型 -> 打包发布 -> 启动验证 -> 清理回收”的顺序整理整个项目的标准流程。目标是让 Windows + MSVC + Qt + LibTorch + CUDA 的构建过程可重复、可检查、可回收。

## 1. 环境准备

### 1.1 必要组件

- Windows 10/11 x64
- Visual Studio 2022 Build Tools 或完整 VS2022
- Qt 6.11.1 MSVC 2022 x64
- CMake 3.30 或兼容版本
- CUDA Toolkit 12.8
- 与 CUDA 匹配的 LibTorch
- Python 3.13（使用系统安装，不创建新的虚拟环境）

### 1.2 推荐目录

- Qt: `D:\Develop\Qt\6.11.1\msvc2022_64`
- LibTorch: `D:\Develop\libtorch`
- 项目: `D:\Develop\Project\Qt\HandwritingRecognition`

### 1.3 环境检查

先确认以下条件成立：

```powershell
py -3.13 --version
```

```powershell
& "D:\Develop\Qt\Tools\CMake_64\bin\cmake.exe" --version
```

```powershell
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" --version
```

### 1.4 隔空书写依赖

- Python 追踪服务使用全局 Python 3.13，不需要也不建议在这个工作区创建新的虚拟环境。
- 需要安装 `mediapipe` 和 `opencv-python`，并在第一次启动隔空书写时允许脚本自动下载 hand landmarker 模型。
- 如果你只做鼠标画板识别和模型训练，这部分依赖不会影响 C++ 主程序构建。

## 2. 清理旧产物

建议在正式构建前先清理一次，避免旧模型、旧包和临时检查目录干扰结果。

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1
```

如果还要清掉 MNIST 原始数据缓存：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1 -IncludeDataCache
```

清理脚本会移除：

- `build/`
- `dist/`
- `artifacts/`
- Python 缓存目录

## 3. 训练模型

当前训练脚本支持 CPU 和 CUDA 两种方式，输出会按来源分类到：

- `artifacts\models\cpu\`
- `artifacts\models\gpu\`

### 3.1 CPU 验证训练

```powershell
py -3.13 scripts\train_mnist.py --epochs 3 --batch-size 256 --output-dir artifacts\models\cpu --device cpu --no-augment --num-workers 0
```

### 3.2 GPU 正式训练

```powershell
py -3.13 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models\gpu --device cuda --pin-memory --num-workers 8
```

如果机器内存或页面文件较紧张，可以优先把 `--num-workers` 降到 `2` 或 `0`。

### 3.3 训练产物

训练完成后，每套模型都应至少包含：

- `mnist_model.pt`
- `model.pth`

## 4. 生成发布包

发布脚本会做三件事：

1. 复制 Qt 可执行依赖。
2. 复制 CPU/GPU 分类模型。
3. 生成一键启动脚本 `run_handwriting_recog.bat`。

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\package_release.ps1 -LibtorchDir D:\Develop\libtorch
```

完成后发布目录为：

```text
D:\Develop\Project\Qt\HandwritingRecognition\dist
```

### 4.0 build 与 dist 的关系

- `build/` 是开发构建目录，存放 CMake/MSBuild 生成的中间文件和 `handwriting_recog.exe`。
- `dist/` 是发布目录，由 `scripts/package_release.ps1` 基于 `build/Release/handwriting_recog.exe` 重新整理得到。
- 这也是为什么项目里会看到两个可执行文件：一个在 `build/Release/`，一个在 `dist/`。前者用于开发调试，后者用于分发和一键运行。
- 如果只重新构建而不执行打包，`build/Release/handwriting_recog.exe` 会更新，但 `dist/` 会保持旧快照。

### 4.1 将构建产物整理成可单独运行的 exe 发布包

本项目的“可单独运行”不是把程序强行压成真正的单文件 exe，而是把 `handwriting_recog.exe`、Qt 运行库、LibTorch DLL 和模型文件整理到同一发布目录，用户双击 exe 就能运行。实际打包时建议按下面的顺序处理：

1. 先完成 Release 构建，确保 `build\Release\handwriting_recog.exe` 已生成。
2. 将 `handwriting_recog.exe` 复制到新的发布目录，例如 `dist\`。
3. 使用 `windeployqt --release --compiler-runtime` 复制 Qt 运行时依赖。
4. 将当前使用的 LibTorch `lib\*.dll` 复制到发布目录，保证推理时能找到 Torch 运行库。
5. 将模型文件按类别放入 `dist\models\cpu` 和 `dist\models\gpu`，每个目录至少包含 `mnist_model.pt` 和 `model.pth`。
6. 生成一个 `run_handwriting_recog.bat` 或类似启动脚本，在启动前把 Qt 和 LibTorch 目录加入 `PATH`，这样用户无需手动配置环境变量。

打包完成后，发布目录中的 exe 可以脱离开发环境直接启动；前提是它所在目录保留了上述依赖文件。若把 exe 单独拷走而不带 Qt/LibTorch DLL，程序将无法正常运行。

## 5. 启动验证

优先使用仓库根目录的一键启动脚本：

```powershell
D:\Develop\Project\Qt\HandwritingRecognition\run_handwriting_recog.bat
```

启动后检查以下内容：

- 主窗口能正常打开。
- 左侧画板可写字。
- 模型下拉框能显示 CPU/GPU 选项。
- CUDA 可用时，GPU 选项可选择。
- 点击“识别”不会出现首次闪退。

## 6. 模型切换规则

- CPU 模型用于兼容和验证。
- GPU 模型用于 CUDA 可用环境下的正式推理。
- 程序会在启动时预热当前模型，减少第一次点击时的延迟和崩溃风险。
- 识别过程中会锁定按钮，防止重复点击重入。

## 7. 发布目录建议保留内容

发布时应保留：

- `dist\handwriting_recog.exe`
- `dist\models\cpu\mnist_model.pt`
- `dist\models\cpu\model.pth`
- `dist\models\gpu\mnist_model.pt`
- `dist\models\gpu\model.pth`
- `dist\run_handwriting_recog.bat`

其他临时检查目录、根目录重复模型文件和日志文件不应再进入发布包。

## 8. 常见问题

### 8.1 点击识别仍然异常

- 确认当前选择的模型文件存在。
- 确认画板里真的有数字，而不是空白页。
- 确认启动日志里显示的推理设备和预期一致。

### 8.2 GPU 选项不显示

- 检查 CUDA 是否可用。
- 检查 LibTorch 是否是 CUDA 版本。
- 检查 `LIBTORCH_DEVICE` 是否被强制设成了 `cpu`。

### 8.3 构建包里缺少模型

- 先检查 `artifacts\models\cpu` 和 `artifacts\models\gpu` 是否完整。
- 再执行一次 `scripts\package_release.ps1`。
