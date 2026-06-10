# 推理设备（CPU / CUDA）工作机制

## 概述

本项目支持两种推理设备：**CPU** 和 **CUDA（NVIDIA GPU）**。项目使用单一模型文件，设备选择通过界面复选框控制，默认使用 CPU。设备决定了模型权重存储在何处、矩阵运算在哪个硬件上执行，两条路径使用完全相同的模型文件和计算图。

## 模型文件

项目只有一份模型文件，不区分设备：

```
artifacts/models/
├── mnist_model.pt    # TorchScript 格式，供 C++ LibTorch 加载
└── model.pth         # PyTorch 原生权重，供 Python 端使用
```

`mnist_model.pt` 是标准的 TorchScript 格式，保存的是网络权重和计算图，**不绑定任何特定硬件**。在 CPU 和 CUDA 设备上都可以加载同一份模型文件：

- 勾选"使用 CUDA 进行推理" → 模型权重加载到 GPU 显存，在 GPU 上执行推理
- 未勾选（默认） → 模型权重加载到 CPU 内存，在 CPU 上执行推理

两种设备下的推理结果完全一致，仅在执行速度上可能存在差异。

## 设备选择机制

设备选择发生在用户界面层，通过 `QCheckBox` 控件控制，逻辑位于 `MainWindow` 中：

```
程序启动
│
├─ 检测 CUDA 可用性：torch::cuda::is_available()
│   ├─ 可用 → 启用复选框，允许用户勾选
│   └─ 不可用 → 禁用复选框，提示"未检测到可用的 CUDA 设备"
│
├─ 默认状态：复选框未勾选 → 使用 CPU
│
└─ 用户交互：
    ├─ 勾选复选框 → 销毁当前识别器 → 以 CUDA 设备重新加载模型
    └─ 取消勾选 → 销毁当前识别器 → 以 CPU 设备重新加载模型
```

设备选择不依赖环境变量或编译宏，完全由运行时的界面交互决定。切换设备时，程序会销毁当前的 `DigitRecognizer` 实例并重新加载模型到新设备上。

`torch::cuda::is_available()` 的检测内容包括：

1. libtorch 是否在编译时启用了 CUDA 支持（是否存在 `torch_cuda.dll`）
2. 系统中是否存在 NVIDIA CUDA 驱动
3. CUDA 运行时能否成功初始化并枚举到可用设备

任意一项不满足，该函数即返回 `false`，复选框将被禁用。

## 模型加载流程

`DigitRecognizer` 构造函数接收两个参数：模型文件路径和是否使用 CUDA。

```
DigitRecognizer::DigitRecognizer(modelPath, useCuda)
│
├─ 1. 文件存在性检查：std::filesystem::exists(modelPath)
│     不存在 → 抛出异常 "Model file not found"
│
├─ 2. 设备决策：
│     ├─ useCuda = true
│     │   ├─ torch::cuda::is_available() = true  → device = kCUDA
│     │   └─ torch::cuda::is_available() = false → 抛出异常
│     └─ useCuda = false → device = kCPU
│
├─ 3. 加载模型：torch::jit::load(modelPath, device)
│     将 TorchScript 模型反序列化，并将权重迁移到目标设备
│     ── CPU 路径：权重加载到 CPU 内存
│     ── CUDA 路径：权重从 CPU 内存拷贝到 GPU 显存
│
└─ 4. 切换推理模式：model.eval()
      关闭 Dropout、将 BatchNorm 切换为使用全局统计量
```

步骤 3 是两条路径的第一个实质差异点。`torch::jit::load` 的第二个参数 `device` 控制权重的目标存放位置。

## CUDA 预热机制

模型加载成功后，若设备为 CUDA，会立即执行一次预热（warmup）：

```
DigitRecognizer::warmUp()
│
├─ device 不是 CUDA → 直接返回（CPU 路径无此步骤）
│
└─ device 是 CUDA：
    ├─ 构造 dummy 输入：torch::zeros({1, 1, 28, 28})，放在 GPU 显存上
    ├─ 执行一次前向推理：model.forward(dummy)
    └─ 同步等待：torch::cuda::synchronize()
```

**为什么 CPU 不需要预热**：CPU 是同步执行设备，第一次计算的延迟与后续计算无显著差异。

**为什么 CUDA 需要预热**：CUDA 采用惰性初始化机制，首次 GPU 操作会触发一系列一次性开销：

| 开销项 | 说明 |
|--------|------|
| CUDA Context 创建 | 驱动与 GPU 之间建立执行上下文 |
| Kernel 编译/缓存 | 将计算图编译为 GPU 可执行的 kernel |
| 显存首次分配 | cuMemAlloc 首次调用的延迟高于后续调用 |
| cuDNN 算法选择 | cuDNN 为特定卷积尺寸搜索最优实现 |

不预热的后果是用户第一次点击"识别"时会感知到明显卡顿。预热将这些开销提前到程序启动阶段（或切换设备时），用户首次推理时即可获得正常响应速度。

## 推理识别流程

以 `predictWithConfidence(const QImage&)` 为例，完整流程如下：

```
输入 QImage（来自画布或摄像头画面）
│
├─ 1. 预处理（preprocessImage）
│     ├─ 转为灰度图（Format_Grayscale8）
│     ├─ 检测墨迹边界，裁剪有效区域
│     ├─ 添加 padding 使区域变为正方形
│     ├─ 缩放到 28×28 像素
│     └─ 归一化：pixel = (1 - raw/255 - mean) / std
│         mean = 0.1307, std = 0.3081（MNIST 数据集统计值）
│     输出：std::vector<float>，长度 784（28×28）
│
├─ 2. 构造输入张量
│     torch::from_blob(data, {1, 1, 28, 28}).clone()
│     形状 [batch=1, channel=1, height=28, width=28]
│     此时张量始终在 CPU 内存上
│
├─ 3. 数据迁移：input.to(device)
│     ├─ CPU 路径：无操作，数据留在 CPU 内存
│     └─ CUDA 路径：数据从 CPU 内存拷贝到 GPU 显存
│
├─ 4. 前向推理（InferenceMode 下执行，关闭梯度计算）
│     model.forward(inputs)
│     ├─ CPU 路径：在 CPU 上执行矩阵乘法、卷积等运算
│     └─ CUDA 路径：在 GPU 上执行，利用数千个 CUDA 核心并行计算
│
├─ 5. 结果回传：logits.to(torch::kCPU)
│     ├─ CPU 路径：无操作，结果已在 CPU
│     └─ CUDA 路径：结果从 GPU 显存拷贝回 CPU 内存
│
└─ 6. 后处理
      softmax(logits) → 取 argmax 得到预测数字
      取对应位置的 softmax 值作为置信度
      返回 PredictResult{digit, confidence}
```

## CPU 与 CUDA 关键差异

| 维度 | CPU | CUDA |
|------|-----|------|
| 权重存储位置 | CPU 内存 | GPU 显存 |
| 计算执行位置 | CPU 核心（少量核心，高主频） | CUDA 核心（数千核心，大规模并行） |
| 数据传输 | 无需额外传输 | 推理前后各一次 CPU↔GPU 拷贝 |
| 预热 | 不需要 | 需要（CUDA 惰性初始化） |
| 适用场景 | 小模型、低延迟单次推理 | 大模型、大输入、批量推理 |
| 可用性 | 始终可用 | 需要 NVIDIA GPU + CUDA 驱动 + cuDNN |

对于本项目的 MNIST 手写数字识别（28×28 灰度图，浅层 CNN），CPU 和 GPU 的推理速度差异可以忽略不计。设备选择机制的设计是通用的，当模型规模增大时，GPU 的并行计算优势会显著体现。
