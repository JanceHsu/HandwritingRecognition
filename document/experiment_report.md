# 手写数字识别实验报告

## 1. 实验背景与意义

手写数字识别是图像分类的经典入门任务，适合验证数据预处理、神经网络训练、模型导出和桌面端推理集成等完整流程。本项目将 PyTorch 训练与 Qt 桌面交互结合，形成“书写-识别-显示”的闭环应用。

## 2. 实验目标

完成一个桌面手写数字识别程序：用户在 Qt 画板上书写数字，系统将画板图像送入 LibTorch 推理模块，返回 0-9 的预测结果。

## 3. 方案设计

整体流程为：Qt 画板采集用户笔迹，图像经本地预处理转换为 28x28 单通道输入，TorchScript 模型完成推理，结果回传到界面显示。

模型采用简单全连接网络：784 -> 512 -> 512 -> 10。

训练阶段使用 PyTorch 训练，并将模型导出为 TorchScript，C++ 侧通过 LibTorch 加载并预测。

## 4. 详细实现

训练脚本位于 `scripts/train_mnist.py`，使用 `torchvision.datasets.MNIST`、`transforms.ToTensor()`、`CrossEntropyLoss` 和 `SGD(lr=1e-3)`。训练完成后保存 `model.pth`，并在 CPU 上 trace 导出 `mnist_model.pt`。

C++ 推理模块位于 `src/recognizer.h` 和 `src/recognizer.cpp`。预处理流程包含灰度转换、缩放到 28x28、归一化到 0-1、反转白底黑字画板输入，并将数据送入 TorchScript 模型。

Qt 界面位于 `src/mainwindow.cpp` 和 `src/canvas.cpp`。画板控件使用白底黑线绘制，提供“识别”和“清空”按钮，并在识别失败或空画布时给出提示。

## 5. 实验结果


- MNIST 测试集准确率（近期 GPU 训练示例）：约 98%（使用小型 CNN + AMP，50 epochs，batch-size=64 在 RTX 5070 上）
- TorchScript 产物：`mnist_model.pt`（放置于 `artifacts/models/{cpu|gpu}`）
- 中间权重：`model.pth`
- 测试图片：`test_images/`
- 桌面程序：发布包位于 `dist/`（包含 `dist/models/cpu` 与 `dist/models/gpu`）

界面截图可在程序运行后自行截取补充；当前工作区已完成可执行文件构建并打包。

## 6. 问题与解决

1. Windows 环境下 `python`、`cmake` 和 `qmake` 起初不在 PATH 中，改用显式路径和 `vcvars64.bat` 完成配置与构建。
2. MNIST 默认下载源速度较慢，训练脚本增加了可配置镜像地址，改用 `https://storage.googleapis.com/cvdf-datasets/mnist/` 后恢复正常。
3. 识别崩溃修复：此前存在本地 OpenCV 兼容层（`src/opencv2/*`）导致的未定义行为（浮点/字节缓冲转换问题），已把识别主路径切换为直接使用 Qt 的 `QImage` 预处理（`src/recognizer.cpp` 新增 `preprocessImage`），并加入：
	- 启动时模型预热（CUDA 时）
	- 识别期间按钮重入保护
	该修复消除了“任意模型点击识别即闪退”的问题。

## 7. 总结与展望

本项目完成了 MNIST 模型训练、TorchScript 导出、Qt 画板界面和 LibTorch 推理集成，形成可运行的桌面应用雏形。后续可继续加入笔迹居中、推理线程化、识别历史记录和更丰富的 UI 美化，以进一步提升实际体验。

附：近期 GPU 训练示例命令（用于复现报告中 98% 结果）

```powershell
py -3.13 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models\gpu --device cuda --pin-memory --num-workers 8
```

备注：若页面文件或系统内存不足，请将 `--num-workers` 降至 2 或 0，以避免 Windows 下的内存分配失败。