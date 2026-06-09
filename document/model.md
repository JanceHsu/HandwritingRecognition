# CPU 模型与 GPU 模型的详细对比说明

本文档基于项目源码 `scripts/train_mnist.py`（训练脚本）和 `src/recognizer.cpp`（C++ 推理模块）的实际实现，详细说明 CPU 模型与 GPU 模型在训练过程、算法策略、优化手段、生成产物以及客户端调用时的全部区别与注意事项。

---

## 1. 网络结构：完全相同

无论是 CPU 模型还是 GPU 模型，使用的网络结构都是同一个 `NeuralNetwork` 类，即一个小型卷积神经网络（Small CNN）。具体结构如下：

**特征提取部分（features）：**
- Conv2d(1, 32, 3×3) → BatchNorm2d → ReLU
- Conv2d(32, 32, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2) → Dropout2d(0.10)
- Conv2d(32, 64, 3×3) → BatchNorm2d → ReLU
- Conv2d(64, 64, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2) → Dropout2d(0.20)

**分类部分（classifier）：**
- Flatten → Linear(64×7×7, 256) → ReLU → Dropout(0.3)
- Linear(256, 128) → ReLU → Dropout(0.2)
- Linear(128, 10)

输入为 1×28×28 的灰度图像，输出为 10 个类别的 logits（对应数字 0-9）。

**结论：** CPU 模型和 GPU 模型的网络层数、参数量、激活函数、Dropout 比例完全一致，区别仅在于训练过程中使用的优化策略和计算设备。

---

## 2. 训练过程的区别

### 2.1 计算设备

| 项目 | CPU 训练 | GPU（CUDA）训练 |
|------|---------|----------------|
| 设备 | `torch.device("cpu")` | `torch.device("cuda")` |
| 启动命令 | `--device cpu` | `--device cuda` |
| 设备检测 | 不检测 | 需要 `torch.cuda.is_available()` 为 True |

脚本中设备选择的逻辑是：当用户指定 `--device cuda` 时，如果 CUDA 不可用则自动回退到 CPU；当用户指定 `--device auto` 或不指定时，脚本会优先使用 CUDA（如果可用），否则回退到 CPU。

### 2.2 内存格式

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| 张量内存布局 | 默认的 NCHW 格式 | `torch.channels_last` 格式 |

GPU 训练时，输入数据和模型都会被转换为 `channels_last` 内存格式。这种格式把通道维度放在最后，更适合 GPU 上的卷积计算，可以提升 cuDNN 卷积 kernel 的执行效率。CPU 训练则使用默认的 NCHW 格式，因为 CPU 上的计算库对 channels_last 没有明显加速。

### 2.3 混合精度训练（AMP）

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| 是否启用 AMP | 否 | 是 |
| 计算精度 | 全部使用 float32 | 前向传播使用 float16，反向传播保持 float32 |
| GradScaler | 不使用 | 使用 `torch.amp.GradScaler("cuda")` |

混合精度训练（Automatic Mixed Precision）是 GPU 训练时的核心优化之一。在 GPU 上，float16 的计算速度大约是 float32 的两倍，同时占用的显存也更少。训练脚本通过 `torch.autocast` 将前向传播中的矩阵乘法和卷积运算自动降为 float16，而反向传播仍使用 float32 来保证梯度精度。

GradScaler 的作用是防止 float16 下梯度太小导致下溢（变成 0）。它会先把 loss 乘以一个较大的缩放系数，计算完梯度后再缩放回去。这个过程对用户完全透明，不需要手动调参。

CPU 训练不使用 AMP，因为 CPU 上 float16 的加速效果不明显，且部分 CPU 指令集对 float16 的支持不如 GPU 完善。

### 2.4 TF32 精度模式

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| TF32 | 不涉及 | 启用（Ampere 及更新架构的 GPU） |

GPU 训练时，脚本会执行以下设置：
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.set_float32_matmul_precision("high")`

TF32 是 NVIDIA Ampere 架构（如 RTX 30 系列、RTX 50 系列等）引入的一种数值格式，它用 19 位来表示浮点数（介于 float16 和 float32 之间），在矩阵乘法中可以接近 float16 的速度，同时保持接近 float32 的精度。对于 MNIST 这种小模型，TF32 的加速幅度有限，但在更大的模型中效果更明显。

### 2.5 cuDNN Benchmark 模式

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| cudnn.benchmark | 不涉及 | 设为 True |

`torch.backends.cudnn.benchmark = True` 让 cuDNN 在第一次遇到某种卷积尺寸时，自动测试多种实现算法并选择最快的。后续再遇到相同尺寸的卷积就直接用选好的算法。这个优化在输入尺寸固定时（如 MNIST 的 28×28）效果很好，但如果输入尺寸频繁变化反而会降低效率。

### 2.6 数据加载器（DataLoader）配置

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| num_workers | 0（默认） | 可配置（如 2 或 8） |
| pin_memory | 不使用 | 可启用 |
| persistent_workers | 不使用 | 启用（当 num_workers > 0） |

- **num_workers**：GPU 训练时可以使用多个子进程并行加载数据，避免数据加载成为瓶颈。CPU 训练通常不设置多 worker，因为 CPU 本身既要训练又要加载数据，多进程反而可能增加开销。在 Windows 上，num_workers 被限制为不超过 2，这是因为 Windows 的多进程启动机制（spawn）比 Linux（fork）更慢，过多 worker 反而可能导致内存问题。
- **pin_memory**：启用后，DataLoader 会把数据放到"锁页内存"（pinned memory）中，这种内存不会被操作系统交换到磁盘，GPU 可以通过 DMA 直接读取，速度比普通内存快。仅在 GPU 训练时有意义。
- **persistent_workers**：让数据加载子进程在整个训练过程中保持存活，避免每个 epoch 重新创建进程的开销。

### 2.7 随机种子

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| CPU 种子 | `torch.manual_seed(42)` | `torch.manual_seed(42)` |
| CUDA 种子 | 不涉及 | `torch.cuda.manual_seed_all(42)` |

两种训练都设置了相同的 CPU 随机种子（42），以保证数据增强中的随机操作（如随机旋转、透视变换等）在多次运行时可复现。GPU 训练额外设置了 CUDA 种子，确保 GPU 上的随机操作（如 Dropout）也具有可复现性。

---

## 3. 数据增强策略：相同

CPU 和 GPU 训练使用完全相同的数据增强流水线（除非用户手动指定 `--no-augment` 关闭）：

- **随机旋转**：±15 度
- **随机仿射变换**：±10 度旋转、10% 平移、0.9-1.1 缩放、±8 度错切
- **随机透视变换**：15% 程度的透视畸变，25% 概率触发
- **颜色抖动**：15% 亮度变化、15% 对比度变化，50% 概率触发
- **归一化**：均值 0.1307、标准差 0.3081（MNIST 数据集的统计量）

这些增强手段的目的都是让模型学会识别各种书写风格的数字，包括倾斜、偏移、大小不一的情况，从而提高实际使用中的识别准确率。

---

## 4. 优化器与学习率调度：相同

两种训练使用完全相同的优化策略：

- **优化器**：AdamW（学习率 1e-3，权重衰减 5e-4）
- **学习率调度**：OneCycleLR
  - 最大学习率：1e-3
  - 预热阶段占比：15%（前 15% 的训练步数逐渐增大学习率）
  - 初始学习率 = 最大学习率 / 10
  - 最终学习率 = 最大学习率 / 1000
- **损失函数**：CrossEntropyLoss，带 label smoothing（系数 0.05）
- **训练轮数**：默认 50 个 epoch
- **批大小**：默认 256（GPU 训练时推荐改为 64）

AdamW 是 Adam 优化器的改进版本，它把权重衰减从梯度更新中分离出来，使得正则化效果更稳定。OneCycleLR 是一种学习率调度策略，先从较小的学习率逐步升高到最大学习率（预热阶段），再逐步降低到很小的学习率。这种"先升后降"的策略有助于模型跳出局部最优解，同时在训练后期稳定收敛。

Label smoothing 是一种正则化手段，它不让模型对训练标签过于"自信"（即不让输出概率过于接近 0 或 1），从而减少过拟合。系数 0.05 表示将 5% 的概率分配给其他类别。

---

## 5. 训练过程本身的区别

虽然优化器和调度器相同，但由于混合精度和 TF32 的存在，GPU 训练的实际行为和 CPU 训练有一些微妙差别：

| 项目 | CPU 训练 | GPU 训练 |
|------|---------|---------|
| 前向传播精度 | 全程 float32 | 部分 float16（通过 autocast） |
| 梯度计算 | 标准反向传播 | 缩放后反向传播（GradScaler） |
| 每步耗时 | 较慢 | 较快（约 2-5 倍加速） |
| 数值精度 | 完全一致的 float32 | 因 float16 引入微小数值差异 |
| 最终准确率 | 约 99.2%-99.5% | 约 99.5%-99.7% |

最终准确率的差异并不完全来自硬件速度，而更多来自训练配置的差异（如批大小、num_workers 等）。理论上，如果 CPU 训练和 GPU 训练使用完全相同的超参数（包括批大小），得到的模型准确率应该非常接近。但实际中，GPU 训练通常使用更大的 batch 配合更多 worker，这会轻微影响最终收敛结果。

---

## 6. 模型导出：完全相同

无论训练时使用 CPU 还是 GPU，模型导出流程完全一致：

```python
# 1. 将模型移到 CPU
model = model.to("cpu")

# 2. 创建示例输入
example = torch.rand(1, 1, 28, 28)

# 3. 使用 TorchScript trace 导出
traced = torch.jit.trace(model.cpu(), example)
traced.save("mnist_model.pt")
```

关键点：
- 导出时**必须**将模型移到 CPU，这样导出的 TorchScript 模型不依赖任何特定设备
- 导出使用的是 `torch.jit.trace`，它通过跟踪一次前向传播的执行路径来生成模型的计算图
- 导出的文件格式为 `.pt`（TorchScript 格式），这是 C++ LibTorch 可以直接加载的格式
- 除了 TorchScript 模型，脚本还会保存 `model.pth`（PyTorch 原生的 state_dict 格式），供 Python 端后续使用或微调

**因此，CPU 模型和 GPU 模型在文件格式上没有任何区别**，都是标准的 TorchScript `.pt` 文件。区别仅在于权重值本身（因为训练过程中的数值精度差异导致最终学到的参数略有不同）。

---

## 7. 客户端调用（C++ 推理）的区别

### 7.1 模型加载与设备选择

C++ 推理模块 `src/recognizer.cpp` 中的设备选择逻辑如下：

1. 读取环境变量 `LIBTORCH_DEVICE`，如果设置了则按其值选择设备
2. 如果未设置环境变量，则使用编译时的宏 `HANDWRITING_RECOG_DEFAULT_DEVICE`（默认为 "auto"）
3. 设备选择规则：
   - `"cpu"` → 固定使用 CPU
   - `"cuda"` → 使用 CUDA（如果不可用则报错）
   - `"auto"` → 优先使用 CUDA（如果可用），否则回退到 CPU

无论加载的是 CPU 模型还是 GPU 模型，C++ 端都可以在 CPU 或 CUDA 上运行推理。这是因为导出的 TorchScript 模型是设备无关的——加载时 LibTorch 会将模型参数放到指定的设备上。

### 7.2 CUDA 预热（WarmUp）

当推理设备为 CUDA 时，识别器会在加载完成后执行一次预热操作：

```cpp
void DigitRecognizer::warmUp()
{
    if (device.type() != torch::kCUDA) return;  // CPU 不预热

    torch::InferenceMode guard;
    auto warmInput = torch::zeros({1, 1, 28, 28}, ...);
    model.forward({warmInput});
    torch::cuda::synchronize();
}
```

预热的目的是：CUDA 的第一次推理会触发 kernel 编译和显存分配，耗时较长（可能几百毫秒甚至超过一秒）。如果不预热，用户第一次点击"识别"按钮时会明显感到卡顿，甚至可能因为超时而被误认为程序崩溃。预热之后，后续推理就只需要几毫秒。

CPU 模型不需要预热，因为 CPU 上不存在 kernel 编译的问题，首次推理和后续推理的耗时基本一致。

### 7.3 推理流程对比

| 步骤 | CPU 推理 | CUDA 推理 |
|------|---------|----------|
| 模型加载 | 加载到 CPU | 加载到 CUDA 设备 |
| 预热 | 跳过 | 执行一次空推理 + synchronize |
| 输入张量 | 创建在 CPU | 创建后 `.to(device)` 移到 GPU |
| 前向传播 | CPU 计算 | GPU 计算 |
| 结果取回 | 直接读取 | `.to(torch::kCPU)` 取回 CPU |
| 单次推理耗时 | 约 1-5ms | 约 1-3ms（预热后） |

从实际体验来看，MNIST 这种小模型在 CPU 和 GPU 上的推理速度差异不大，因为模型本身参数量很少（约 20 万个参数），数据传输的开销可能比计算本身还大。GPU 的优势主要体现在大批量推理或更大模型的场景中。

### 7.4 图像预处理：完全相同

无论使用 CPU 模型还是 GPU 模型，图像预处理流程完全一致：

1. **转灰度图**：如果是彩色图像（3 通道或 4 通道），先转为单通道灰度图
2. **寻找笔迹边界**：扫描所有像素，找到灰度值低于 245 的"墨迹"区域的边界框
3. **边距扩展**：在边界框四周各扩展 4 个像素的边距
4. **居中裁切**：将笔迹区域裁切为正方形（以较长边为准），白色填充短边方向的空白
5. **缩放到 28×28**：使用 `cv::INTER_AREA` 插值算法缩放
6. **归一化**：像素值从 [0, 255] 映射到 [0, 1]，然后反转（因为画板是白底黑字，MNIST 是黑底白字），最后减去均值 0.1307 并除以标准差 0.3081

这些预处理步骤确保用户在画板上写的任何位置、任何大小的数字，都能被转换成和 MNIST 训练数据分布一致的 28×28 图像。

### 7.5 Qt 界面中的模型切换

在 Qt 主界面中，用户可以通过下拉框选择"CPU 模型"或"GPU 模型（推荐）"：

- **GPU 模型**：默认选项（如果 CUDA 可用），加载 `models/gpu/mnist_model.pt`
- **CPU 模型**：备选选项，加载 `models/cpu/mnist_model.pt`
- 如果系统检测到 CUDA 不可用，GPU 选项会被移除，只保留 CPU 模型

切换模型时，程序会销毁当前的识别器并重新创建，同时执行预热。操作日志会记录模型加载状态和使用的推理设备。

**注意事项**：下拉框中的"CPU 模型"和"GPU 模型"指的是训练来源（即用 CPU 还是 GPU 训练的），而不是推理设备。即使选择了"GPU 模型"，如果当前系统没有 CUDA，推理仍然会在 CPU 上执行。反过来，"CPU 模型"也可以在有 CUDA 的系统上用 GPU 推理。

### 7.6 模型文件查找与回退机制

程序启动时会按以下顺序查找模型文件：

1. 优先查找 `models/{key}/mnist_model.pt`（key 为 "cpu" 或 "gpu"）
2. 搜索路径包括：可执行文件目录、上级目录、`artifacts/models`、`dist/models` 等多个候选位置
3. 如果找不到用户选择的模型，会自动回退到另一种模型（如找不到 gpu 模型则尝试加载 cpu 模型）
4. 如果所有模型都找不到，识别功能将被禁用，界面给出提示

---

## 8. 典型训练命令对比

### CPU 训练（快速验证用）

```powershell
py -3.13 scripts\train_mnist.py ^
    --epochs 3 ^
    --batch-size 256 ^
    --output-dir artifacts\models\cpu ^
    --device cpu ^
    --no-augment ^
    --num-workers 0
```

特点：关闭数据增强、不用多 worker，适合快速验证代码是否能正常运行。

### GPU 训练（正式训练用）

```powershell
py -3.13 scripts\train_mnist.py ^
    --epochs 50 ^
    --batch-size 64 ^
    --output-dir artifacts\models\gpu ^
    --device cuda ^
    --pin-memory ^
    --num-workers 8
```

特点：开启全部优化，使用小 batch（64）配合更多 worker，训练 50 个 epoch 以获得最佳准确率。

---

## 9. 总结

| 维度 | CPU 模型 | GPU 模型 |
|------|---------|---------|
| 网络结构 | 小型 CNN（完全相同） | 小型 CNN（完全相同） |
| 数据增强 | 旋转、仿射、透视、颜色抖动 | 旋转、仿射、透视、颜色抖动 |
| 优化器 | AdamW + OneCycleLR | AdamW + OneCycleLR |
| 训练精度 | 全程 float32 | float16 混合精度 + TF32 |
| 训练速度 | 较慢 | 较快（2-5 倍） |
| 最终准确率 | 约 99.2%-99.5% | 约 99.5%-99.7% |
| 导出格式 | TorchScript .pt（CPU trace） | TorchScript .pt（CPU trace） |
| C++ 推理设备 | CPU 或 CUDA 均可 | CPU 或 CUDA 均可 |
| CUDA 预热 | 不需要 | 首次推理前预热 |
| 推理速度 | 约 1-5ms | 约 1-3ms（预热后） |
| 图像预处理 | 完全相同 | 完全相同 |
| 适用场景 | 无 GPU 的电脑、快速验证 | 有 NVIDIA GPU 的电脑、追求高精度 |

简单来说，CPU 模型和 GPU 模型本质上是同一个网络在不同条件下训练出来的。GPU 训练利用了混合精度和各种硬件加速技巧，所以训练更快、最终精度略高。但在客户端调用时，两者的使用方式几乎完全相同，区别仅在于 CUDA 预热和推理设备的选择。
