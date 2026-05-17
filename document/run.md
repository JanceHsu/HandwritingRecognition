## 手写数字识别项目运行说明

本文档说明本项目在 Windows + MSVC + Qt 6.11.1 环境下，使用 LibTorch 2.7.1（默认 CPU，可切换 CUDA）的编译、打包和运行方式。

### 1. 环境要求

- Windows 10/11
- Visual Studio 2022 Build Tools 或 Visual Studio 2022 Community
- Qt 6.11.1 MSVC 64-bit
- LibTorch 2.7.1（CPU 或 CUDA）
- Python 3.13

本项目当前使用的路径示例：

- Qt: `D:\Develop\Qt\6.11.1\msvc2022_64`
- LibTorch(默认): `D:\Develop\libtorch`
- 工程根目录: `D:\Develop\Project\Qt\HandwritingRecognition`

### 2. 编译步骤

先打开一个命令行窗口，建议直接使用 `cmd`，然后执行：

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

随后配置和构建项目：

```bat
cmake -S D:\Develop\Project\Qt\HandwritingRecognition -B D:\Develop\Project\Qt\HandwritingRecognition\build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="D:/Develop/Qt/6.11.1/msvc2022_64;D:/Develop/libtorch" -DTorch_DIR="D:/Develop/libtorch/share/cmake/Torch" -DDIGIT_RECOG_DEFAULT_DEVICE=cpu
cmake --build D:\Develop\Project\Qt\HandwritingRecognition\build --config Release
```

如需切换到 CUDA 版 LibTorch，可先执行准备脚本：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\prepare_libtorch_cuda.ps1 -CudaLibtorchDir D:\Develop\libtorch-cuda
```

然后将配置命令中的 LibTorch 路径改为 CUDA 目录，并将默认设备改为 auto 或 cuda。

编译成功后，可执行文件位于：

```text
D:\Develop\Project\Qt\HandwritingRecognition\build\Release\digit_recog.exe
```

### 3. 训练模型

训练脚本会自动从 MNIST 镜像下载数据并导出模型文件（默认输出到 `artifacts/`）。当前脚本使用小型 CNN + CUDA AMP，通常比纯 MLP 更容易把 GPU 跑起来，也能获得更高准确率。

GPU 训练示例（短轮次用于验证）:

```powershell
py -3.13 scripts\train_mnist.py --epochs 2 --batch-size 256 --output-dir artifacts --device cuda --pin-memory --num-workers 2
```

完整训练示例（在 CUDA 上，将 epochs 设为所需轮数，例如 50）:

```powershell
py -3.13 scripts\train_mnist.py --epochs 50 --batch-size 256 --output-dir artifacts --device cuda --pin-memory --num-workers 2
```

完成后会生成：

- `artifacts\model.pth`
- `artifacts\mnist_model.pt`  (TorchScript, CPU-traced for C++ 兼容性)

如果 Windows 机器页面文件较小，`--num-workers` 可以进一步降到 `0`，但优先建议先尝试 `2`。

当前推荐把模型按训练来源分类存放：

- `artifacts\models\cpu\mnist_model.pt`
- `artifacts\models\gpu\mnist_model.pt`

窗口里会显示 `CPU 模型` 和 `GPU 模型（推荐）` 两个选项，CUDA 可用时才允许选择 GPU 模型。

### 4. 打包发布版

打包脚本会复制 Qt 运行库、LibTorch 运行库和模型文件到 `dist` 目录，并自动生成一键启动脚本。

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\package_release.ps1
```

打包完成后，发布目录如下：

```text
D:\Develop\Project\Qt\HandwritingRecognition\dist
```

### 5. 清理项目

如果需要回到一个干净的工作区，优先使用清理脚本而不是手动删除文件：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1
```

如需连同 MNIST 下载缓存一起清除：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1 -IncludeDataCache
```

### 6. 模型选择与运行说明

- `artifacts\models\cpu\mnist_model.pt` 是 CPU 模型。
- `artifacts\models\gpu\mnist_model.pt` 是 GPU 模型。
- 程序启动后会根据 CUDA 可用性控制 GPU 选项是否可选。
- 识别按钮采用启动预热后的单次推理路径，避免首次点击时出现长时间卡顿或闪退。

### 7. 推荐的完整构建顺序

1. 配置 Qt 6.11.1 MSVC 和 LibTorch CUDA 路径。
2. 训练 CPU 或 GPU 模型，输出到 `artifacts\models\cpu` 和 `artifacts\models\gpu`。
3. 运行 `scripts\package_release.ps1` 生成 `dist`。
4. 使用 `run_digit_recog.bat` 启动程序并确认模型切换与识别正常。

目录中应包含：

- `digit_recog.exe`
- `run_digit_recog.bat`
- `Qt6Core.dll`、`Qt6Gui.dll`、`Qt6Widgets.dll`
- `torch_cpu.dll`、`torch.dll`、`c10.dll`
- `mnist_model.pt`

如果使用 CUDA 版 LibTorch，发布目录中的 Torch 相关 DLL 会由打包脚本按 `LibtorchDir` 自动复制。

### 5. 一键运行（推荐唯一入口）

推荐直接双击工程根目录中的 `run_digit_recog.bat`，这是唯一推荐入口。该脚本会在缺少发布包时自动调用打包流程：

```text
D:\Develop\Project\Qt\HandwritingRecognition\run_digit_recog.bat
```

### 6. 程序内日志查看

程序主窗口右侧增加了“操作日志”区域，会显示：

- 程序启动
- 模型加载成功或失败
- 点击识别
- 画板是否为空
- 识别结果
- 识别异常信息

### 7. 常见问题

#### 7.1 启动后提示入口点错误

这通常说明加载到了错误版本的 Qt DLL。必须优先使用 `msvc2022_64\bin` 下的 Qt 运行库。

#### 7.4 如何指定 CPU/CUDA 设备

程序日志里的 `推理设备=cpu` 只表示当前模型在 CPU 上做推理，不表示模型不是 GPU 训练出来的。当前训练脚本已经是 CUDA 训练，默认启动器会优先用 `auto`，有 CUDA 就走 GPU。

可通过环境变量覆盖推理设备：

```powershell
$env:LIBTORCH_DEVICE = "cpu"   # 可选值: cpu / cuda / auto
```

可同时指定 LibTorch 目录（例如 CUDA 版本）：

```powershell
$env:LIBTORCH_DIR = "D:\Develop\libtorch-cuda"
$env:LIBTORCH_DEVICE = "cuda"
D:\Develop\Project\Qt\HandwritingRecognition\run_digit_recog.bat
```

#### 7.2 点击识别仍然退出

请先查看右侧“操作日志”区域，确认最后一条记录停在哪一步。若仍然退出，优先检查模型文件是否存在，以及 `dist` 目录中是否包含完整的 Qt 和 LibTorch DLL。

#### 7.3 模型加载失败

确认 `mnist_model.pt` 已放在程序当前工作目录或 `dist` 目录中，并且文件没有被移动或改名。

### 8. 推荐工作流

最稳妥的日常工作流是：

1. 修改代码
2. 重新编译
3. 重新打包
4. 双击 `run_digit_recog.bat`

这样可以避免手工调整 PATH 引发的 DLL 混用问题。

### 9. 项目清理

清理构建产物与缓存目录（`build`、`dist`、`scripts\__pycache__`）：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1
```

如果还需要清理数据缓存与测试导出图（`data\MNIST\raw`、`test_images`），可执行：

```powershell
powershell -ExecutionPolicy Bypass -File D:\Develop\Project\Qt\HandwritingRecognition\scripts\clean_project.ps1 -IncludeDataCache
```
