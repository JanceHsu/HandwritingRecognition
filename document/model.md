# 模型训练、推理与设备管理

本文档基于项目源码 `scripts/train_mnist.py`（训练脚本）和 `src/recognizer.cpp`（C++ 推理模块）的实际实现，说明模型在训练、导出、推理和设备切换方面的完整技术细节。

---

## 1. 网络结构

项目使用单一模型文件，CPU 和 CUDA 推理加载同一份 `mnist_model.pt`。网络结构为小型卷积神经网络（Small CNN）：

**特征提取部分（features）：**
- Conv2d(1, 32, 3x3) -> BatchNorm2d -> ReLU
- Conv2d(32, 32, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2) -> Dropout2d(0.10)
- Conv2d(32, 64, 3x3) -> BatchNorm2d -> ReLU
- Conv2d(64, 64, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2) -> Dropout2d(0.20)

**分类部分（classifier）：**
- Flatten -> Linear(64x7x7, 256) -> ReLU -> Dropout(0.3)
- Linear(256, 128) -> ReLU -> Dropout(0.2)
- Linear(128, 10)

输入为 1x28x28 的灰度图像，输出为 10 个类别的 logits（对应数字 0-9）。

---

## 2. 训练过程

### 2.1 数据准备与增强

训练使用 MNIST 数据集（60000 张训练图片，10000 张测试图片）。训练脚本对数据进行以下增强：

- 随机旋转：+-15 度
- 随机仿射变换：+-10 度旋转、10% 平移、0.9-1.1 缩放、+-8 度错切
- 随机透视变换：15% 程度畸变，25% 概率触发
- 颜色抖动：15% 亮度和对比度变化，50% 概率触发
- 归一化：均值 0.1307，标准差 0.3081（MNIST 数据集统计量）

### 2.2 优化器与学习率调度

- **优化器**：AdamW（学习率 1e-3，权重衰减 5e-4）
- **学习率调度**：OneCycleLR
  - 最大学习率：1e-3
  - 预热阶段占比：15%（前 15% 的训练步数逐渐增大学习率）
  - 初始学习率 = 最大学习率 / 10
  - 最终学习率 = 最大学习率 / 1000
- **损失函数**：CrossEntropyLoss，带 label smoothing（系数 0.05）
- **训练轮数**：默认 50 个 epoch
- **批大小**：默认 256（GPU 训练时推荐改为 64）

### 2.3 CPU 训练与 GPU 训练的区别

训练脚本支持 CPU 和 CUDA 两种方式，网络结构和数据增强完全相同。区别在于：

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| 混合精度（AMP） | 不启用 | 启用，前向传播用 float16，反向传播保持 float32 |
| TF32 精度 | 不涉及 | 启用（Ampere 及更新架构） |
| cuDNN benchmark | 不涉及 | 启用，自动选择最快卷积算法 |
| 数据加载 | num_workers=0 | 可配置（Windows 上限 2），支持 pin_memory |
| 训练速度 | 较慢 | 较快（约 2-5 倍加速） |
| 最终准确率 | 约 99.2%-99.5% | 约 99.5%-99.7% |

GPU 训练时的全局优化设置：

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

GPU 模式下的 DataLoader 配置：

```python
dl_kwargs = {}
if device.type == "cuda":
    safe_num_workers = max(0, args.num_workers)
    if os.name == "nt":  # Windows 下限制 worker 数量
        safe_num_workers = min(safe_num_workers, 2)
    dl_kwargs = {
        "num_workers": safe_num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": safe_num_workers > 0,
    }
```

---

## 3. 模型导出

无论训练时使用 CPU 还是 GPU，导出流程完全一致，且**导出时必须将模型移到 CPU**：

```python
# 保存 PyTorch 原生权重
torch.save(model.state_dict(), "model.pth")

# 导出 TorchScript 模型（必须在 CPU 上 trace）
model = model.to("cpu")
example = torch.rand(1, 1, 28, 28)
traced = torch.jit.trace(model.cpu(), example)
traced.save("mnist_model.pt")
```

导出产物：
- `model.pth`：PyTorch 原生 state_dict 格式，供 Python 端使用
- `mnist_model.pt`：TorchScript 格式，供 C++ LibTorch 加载

两个文件都不绑定特定设备，在 CPU 和 CUDA 上均可加载。

---

## 4. 推理设备管理

### 4.1 设备选择机制

程序通过界面复选框控制推理设备，逻辑位于 `MainWindow`：

```
程序启动
├─ 检测 CUDA 可用性：torch::cuda::is_available()
│   ├─ 可用 -> 启用复选框
│   └─ 不可用 -> 禁用复选框，提示"未检测到可用的 CUDA 设备"
├─ 默认状态：复选框未勾选 -> 使用 CPU
└─ 用户交互：
    ├─ 勾选复选框 -> 销毁当前识别器 -> 以 CUDA 重新加载模型
    └─ 取消勾选 -> 销毁当前识别器 -> 以 CPU 重新加载模型
```

`torch::cuda::is_available()` 的检测内容包括：
1. LibTorch 是否在编译时启用了 CUDA 支持（是否存在 `torch_cuda.dll`）
2. 系统中是否存在 NVIDIA CUDA 驱动
3. CUDA 运行时能否成功初始化并枚举到可用设备

任意一项不满足，该函数即返回 `false`，复选框将被禁用。

### 4.2 模型加载流程

`DigitRecognizer` 构造函数接收模型路径和是否使用 CUDA 两个参数：

```
DigitRecognizer(modelPath, useCuda)
├─ 1. 文件存在性检查
├─ 2. 设备决策：
│   ├─ useCuda = true -> torch::cuda::is_available() 检查 -> device = kCUDA
│   └─ useCuda = false -> device = kCPU
├─ 3. 加载模型：torch::jit::load(modelPath, device)
│     将权重迁移到目标设备（CPU 内存 或 GPU 显存）
└─ 4. model.eval()：关闭 Dropout、切换 BatchNorm 为全局统计量
```

### 4.3 CUDA 预热机制

模型加载成功后，若设备为 CUDA，立即执行一次预热：

```cpp
void DigitRecognizer::warmUp()
{
    if (device.type() != torch::kCUDA) return;

    torch::InferenceMode guard;
    auto warmInput = torch::zeros({1, 1, 28, 28},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(warmInput);
    (void)model.forward(inputs).toTensor();
    torch::cuda::synchronize();
}
```

CUDA 需要预热的原因：首次 GPU 操作会触发一系列一次性开销：

| 开销项 | 说明 |
|--------|------|
| CUDA Context 创建 | 驱动与 GPU 之间建立执行上下文 |
| Kernel 编译/缓存 | 将计算图编译为 GPU 可执行的 kernel |
| 显存首次分配 | cuMemAlloc 首次调用的延迟高于后续调用 |
| cuDNN 算法选择 | cuDNN 为特定卷积尺寸搜索最优实现 |

CPU 不需要预热，因为 CPU 是同步执行设备，首次计算的延迟与后续无显著差异。

---

## 5. 图像预处理

### 5.1 预处理流程

无论使用 CPU 还是 CUDA 推理，图像预处理完全一致。以 `preprocess(const cv::Mat&)` 为例：

1. **转灰度图**：3 通道或 4 通道转为单通道灰度
2. **寻找笔迹边界**：逐像素扫描灰度值 < 245 的区域，确定边界框
3. **边距扩展与居中裁切**：边界框四周扩展 4 像素，裁切为正方形（白色填充短边方向）
4. **缩放到 28x28**：使用 `cv::INTER_AREA` 插值算法
5. **归一化**：`pixel = (1 - raw/255 - 0.1307) / 0.3081`（反转 + 减均值除标准差）

QImage 版本的 `preprocessImage(const QImage&)` 使用纯 Qt API 实现相同的逻辑。

### 5.2 推理流程

以 `predictWithConfidence(const QImage&)` 为例：

```
输入 QImage
├─ 1. 预处理 -> std::vector<float>(784)
├─ 2. 构造张量：torch::from_blob(data, {1, 1, 28, 28}).clone()
├─ 3. 数据迁移：input.to(device)
├─ 4. 前向推理（InferenceMode 下，关闭梯度计算）
├─ 5. 结果回传：logits.to(torch::kCPU)
└─ 6. softmax -> argmax -> PredictResult{digit, confidence}
```

---

## 6. 典型训练命令

### CPU 验证训练

```powershell
py -3.12 scripts\train_mnist.py --epochs 3 --batch-size 256 --output-dir artifacts\models\cpu --device cpu --no-augment --num-workers 0
```

### GPU 正式训练

```powershell
py -3.12 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models\gpu --device cuda --pin-memory --num-workers 8
```

若系统内存或页面文件较紧张，可将 `--num-workers` 降至 2 或 0。

---

## 7. CPU 与 CUDA 关键差异

| 维度 | CPU | CUDA |
|------|-----|------|
| 权重存储位置 | CPU 内存 | GPU 显存 |
| 计算执行位置 | CPU 核心 | CUDA 核心（大规模并行） |
| 数据传输 | 无需额外传输 | 推理前后各一次 CPU-GPU 拷贝 |
| 预热 | 不需要 | 需要（CUDA 惰性初始化） |
| 单次推理耗时 | 约 1-5ms | 约 1-3ms（预热后） |
| 适用场景 | 通用 | 大模型、大批量推理 |

对于本项目的 MNIST 小型 CNN（约 20 万参数），CPU 和 GPU 推理速度差异不大，因为模型计算量小，数据传输开销可能比计算本身还大。

---

## 8. 实际训练结果

在 RTX 5070 上使用 CUDA 进行 50 个 epoch 的完整训练后，模型在 MNIST 测试集上的准确率约为 99.7%。训练命令：

```powershell
py -3.12 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models\gpu --device cuda --pin-memory --num-workers 8
```

若页面文件或系统内存不足，请将 `--num-workers` 降至 2 或 0，以避免 Windows 下的内存分配失败。
