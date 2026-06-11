# 基于深度学习的手写数字识别系统——项目报告

---

## 一、项目背景

手写数字识别是计算机视觉领域最经典的入门问题之一。它的任务很明确：给定一张手写数字的图片，让计算机判断出写的是 0 到 9 中的哪一个。这个问题之所以被广泛使用，是因为它足够简单，可以用较短的时间跑通"数据准备 → 模型训练 → 部署使用"的完整流程，同时又足够有意义——邮政编码自动识别、银行支票数字录入、快递单号扫描等场景都依赖类似的技术。

本项目的目标不仅仅是训练出一个高精度的识别模型，而是要把模型真正做成一个普通用户能直接使用的桌面软件。具体来说，用户打开程序后，可以用鼠标在画板上写一个数字，程序会立刻告诉他识别结果是什么。在这个基础功能之上，我们还进一步实现了"隔空书写"：用户对着摄像头伸出食指在空中比划，程序通过手部追踪技术捕捉指尖运动，将其映射到画布上，从而实现不用接触屏幕也能写字并识别。

这样的设计让项目从一个单纯的机器学习练习，变成了一个涉及 Python 训练、C++ 推理、图形界面开发、摄像头实时处理、多进程通信等多个技术领域的综合工程实践。

---

## 二、项目目标

本项目的具体目标分为四个方面：

**目标一：模型训练与导出。** 使用 Python 和 PyTorch 在 MNIST 数据集上训练一个手写数字分类模型，将训练好的模型导出为 TorchScript 格式，使 C++ 端可以直接加载使用。要求模型在测试集上的准确率达到 99% 以上。

**目标二：桌面端识别应用。** 使用 Qt 框架开发一个桌面程序，包含手写画布、识别按钮、结果展示、推理设备切换等基本功能。用户在画布上书写数字后，程序能实时完成识别并显示结果。

**目标三：隔空书写功能。** 在基本识别功能之上，通过摄像头实时捕捉用户的手部运动，让用户可以在空中"书写"数字。这个功能需要跨语言协作——Qt 端负责摄像头采集和画面显示，Python 端负责手部关键点检测，两者通过进程间通信交换数据。

**目标四：工程化构建与发布。** 将整个项目的构建、打包、启动流程固定下来，使得项目可以在 Windows + MSVC 环境下重复构建，并生成可直接分发的发布包。

下表列出了本项目使用的主要技术栈及其作用：

| 模块 | 技术栈 | 作用 |
|------|--------|------|
| 模型训练 | Python 3.12, PyTorch, Torchvision, CUDA 12.8 | 训练小型 CNN 模型并导出 TorchScript |
| 模型推理 | LibTorch, TorchScript, QImage | 在 C++ 端加载模型，对画布图像进行预处理后返回识别结果 |
| 图形界面 | Qt 6 Widgets | 实现用户界面，包含画布、按钮、推理设备切换、操作日志 |
| 手部追踪 | OpenCV, MediaPipe, QProcess | 通过摄像头检测手部关键点，驱动隔空书写 |
| 工程构建 | CMake, MSVC 2022 | 标准化构建流程，支持 Debug/Release 配置 |

---

## 三、总体方案设计

### 3.1 系统架构

整个系统由三个主要部分组成，它们之间的关系可以用一条数据流来描述：

用户书写 → 画布采集图像 → 预处理（缩放、归一化等） → 模型推理 → 显示识别结果

**训练端（Python）：** 负责模型的训练和导出。脚本 `scripts/train_mnist.py` 从 MNIST 数据集加载手写数字图片，经过数据增强后送入卷积神经网络训练，最终将模型导出为 TorchScript 格式的 `.pt` 文件。

**推理端（C++ + LibTorch）：** 负责加载训练好的模型，对用户输入的图像进行预处理，然后执行推理得到识别结果。推理模块封装在 `DigitRecognizer` 类中，对外只暴露 `predict()` 接口。

**界面端（Qt Widgets）：** 负责和用户交互。画板控件 `Canvas` 接收鼠标或摄像头输入，主窗口 `MainWindow` 管理按钮事件、推理设备切换、结果显示和操作日志。

对于隔空书写功能，系统还需要一个额外的协作链路：

摄像头画面（Qt 采集） → JPEG 压缩 → 通过 stdin 发送给 Python 进程 → MediaPipe 手部检测 → JSON 结果通过 stdout 返回 → Qt 映射到画布

这个设计的核心思路是：Qt 只负责"看"（摄像头采集和画面显示），Python 只负责"算"（手部关键点检测），两者之间只传递轻量的坐标数据，不传输图像，从而避免管道阻塞和延迟问题。

### 3.2 模型结构

项目使用的网络是一个小型卷积神经网络（Small CNN），由两个部分组成：

**特征提取部分：** 由四层卷积层、批归一化层、ReLU 激活函数和最大池化层组成。前两层卷积使用 32 个卷积核，后两层使用 64 个卷积核。两次最大池化将特征图从 28×28 缩小到 7×7。两个 Dropout 层（概率分别为 0.10 和 0.20）用于防止过拟合。

**分类部分：** 将特征图展平后，通过三层全连接层（64×7×7 → 256 → 128 → 10）逐步将特征映射到 10 个类别上。同样使用 Dropout（概率 0.3 和 0.2）进行正则化。

相比早期版本使用的简单全连接网络（784 → 512 → 512 → 10），卷积神经网络能更好地捕捉图像中的局部结构特征（如笔画的边缘、转角等），因此在同等训练轮数下准确率更高。

以下是 `train_mnist.py` 中的网络定义：

```python
class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 第1层卷积
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 第2层卷积
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                               # 28×28 → 14×14
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 第3层卷积
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第4层卷积
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                               # 14×14 → 7×7
            nn.Dropout2d(0.20),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10),           # 输出 10 个类别
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
```

---

## 四、详细实现

### 4.1 模型训练

#### 4.1.1 数据准备

训练使用 MNIST 数据集，包含 60000 张训练图片和 10000 张测试图片，每张图片都是 28×28 像素的灰度手写数字。

为了让模型能识别各种书写风格，训练脚本对训练数据进行了丰富的数据增强处理：

- 随机旋转：每张图片随机旋转 ±15 度
- 随机仿射变换：包括 ±10 度的额外旋转、10% 的平移、0.9 到 1.1 的缩放、±8 度的错切
- 随机透视变换：以 25% 的概率对图片进行轻微的透视畸变
- 颜色抖动：以 50% 的概率随机调整图片的亮度和对比度

这些增强手段模拟了真实手写中可能出现的各种情况——字写歪了、写偏了、写大了写小了、或者纸张不平整导致的变形。经过增强后，模型对不同书写风格的适应能力会显著提高。

以下是 `train_mnist.py` 中实际的数据增强与加载代码：

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.10, 0.10),
        scale=(0.90, 1.10),
        shear=8,
    ),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
mnist_train = datasets.MNIST(
    root=str(data_root), train=True,
    transform=train_transform, download=True
)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
```

最后，所有图片都会被归一化：像素值除以 255 转为 0 到 1 的浮点数，再减去 MNIST 数据集的均值（0.1307）并除以标准差（0.3081）。这一步的目的是让输入数据的分布更"规整"，有助于模型更快、更稳定地收敛。

#### 4.1.2 训练策略

训练使用了以下策略：

**优化器选择 AdamW，** 学习率 0.001，权重衰减 0.0005。AdamW 是 Adam 优化器的改进版本，它把"权重衰减"（一种防止模型过拟合的正则化手段）从梯度更新中独立出来，使得正则化效果更可控。

**学习率调度使用 OneCycleLR。** 这种策略让学习率先从一个较小值逐渐升高到最大学习率（前 15% 的训练步数用于这个"预热"阶段），然后再逐渐降低到很小的值。这种"先升后降"的节奏有助于模型在训练初期快速探索，后期精细调整。

**损失函数使用带标签平滑的交叉熵。** 标签平滑（label smoothing，系数 0.05）是一种正则化手段，它不让模型对训练标签过于"自信"，从而减少过拟合。具体来说，它将 5% 的概率分配给错误的类别，让模型的输出不要太极端。

以下是 `train_mnist.py` 中对应的训练策略代码：

```python
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    epochs=args.epochs,
    steps_per_epoch=max(1, len(train_loader)),
    pct_start=0.15,       # 前 15% 步数用于预热
    div_factor=10.0,       # 初始学习率 = max_lr / 10
    final_div_factor=100.0 # 最终学习率 = max_lr / 1000
)
```

在每个训练批次（batch）中，学习率调度器都会更新一次，整个训练循环的核心逻辑如下：

```python
def train(dataloader, model, loss_fn, optimizer, device, scaler=None, scheduler=None):
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
```

#### 4.1.3 CPU 训练与 GPU 训练的区别

训练脚本同时支持 CPU 和 GPU 两种训练方式。两者使用完全相同的网络结构和数据增强策略，区别在于：

**GPU 训练启用了混合精度训练（AMP）。** 混合精度训练的意思是，前向传播中的一部分计算（主要是矩阵乘法和卷积）使用 float16（半精度浮点数）而不是 float32（单精度浮点数），float16 的计算速度大约是 float32 的两倍，同时占用的显存也更少。但 float16 的精度较低，梯度可能太小而被"吞掉"，所以反向传播仍然使用 float32，同时用 GradScaler 先放大 loss 再计算梯度，算完后再缩放回去，避免精度损失。

以下是 `train_mnist.py` 中 GPU 训练时的全局优化设置和混合精度训练代码：

```python
# ── GPU 全局优化设置（仅在 CUDA 可用时生效）──
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True            # 自动选择最快卷积算法
    torch.set_float32_matmul_precision("high")       # 允许 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

```python
# ── 训练批次中的混合精度逻辑 ──
use_amp = scaler is not None and device.type == "cuda"
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)

if scaler is not None and device.type == "cuda":
    scaler.scale(loss).backward()  # 放大 loss → 计算梯度
    scaler.step(optimizer)         # 缩放梯度 → 更新参数
    scaler.update()                # 调整缩放系数
else:
    loss.backward()
    optimizer.step()
```

**GPU 训练启用了 TF32 精度模式。** 这是 NVIDIA 新一代 GPU（如 RTX 30 系列及以上）的特性，用 19 位浮点数代替 32 位，在矩阵乘法中可以获得接近 float16 的速度和接近 float32 的精度。

**GPU 训练启用了 cuDNN benchmark。** 让 CUDA 的深度学习库自动测试多种卷积实现算法并选择最快的，后续遇到相同尺寸的卷积直接用选好的算法。

**GPU 训练的数据加载器可以使用多进程和锁页内存。** 多进程（num_workers）让数据加载和模型训练并行进行，不互相等待；锁页内存（pin_memory）让数据传输到 GPU 时走 DMA 通道，速度更快。对应的 DataLoader 配置代码如下：

```python
# ── GPU 模式下重建 DataLoader，加入性能优化参数 ──
dl_kwargs = {}
if device.type == "cuda":
    safe_num_workers = max(0, args.num_workers)
    if os.name == "nt":                # Windows 下限制 worker 数量
        safe_num_workers = min(safe_num_workers, 2)
    dl_kwargs = {
        "num_workers": safe_num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": safe_num_workers > 0,
    }
train_loader = DataLoader(
    train_loader.dataset, batch_size=args.batch_size,
    shuffle=True, **dl_kwargs
)
```

这些优化使得 GPU 训练速度通常是 CPU 的 2 到 5 倍。在最终精度上，GPU 训练通常能达到 99.5% 以上（因为可以训练更多轮次），CPU 训练在较短轮次下通常在 99.2% 左右。

不过有一个关键点：**无论训练时用的是 CPU 还是 GPU，导出的模型文件格式完全相同。** 导出时脚本会把模型移到 CPU 上，用一个随机输入执行一次前向传播，通过 `torch.jit.trace` 将执行过程记录下来，生成标准的 TorchScript `.pt` 文件。这个文件不依赖任何特定设备，在 C++ 端可以加载到 CPU 或 GPU 上运行。

导出相关代码如下：

```python
# 保存 PyTorch 原生权重
model_path = args.output_dir / "model.pth"
torch.save(model.state_dict(), model_path)

# 导出 TorchScript 模型（必须在 CPU 上 trace）
model = model.to("cpu")
example = torch.rand(1, 1, 28, 28)          # 与 MNIST 输入尺寸一致
traced = torch.jit.trace(model.cpu(), example)
traced_path = args.output_dir / "mnist_model.pt"
traced.save(str(traced_path))
```

#### 4.1.4 训练结果

在 GPU 上进行 50 个 epoch 的完整训练后，模型在 MNIST 测试集上的准确率约为 99.7%。训练产物包括两个文件：

- `model.pth`：PyTorch 原生格式的模型权重，供 Python 端后续使用
- `mnist_model.pt`：TorchScript 格式的模型，供 C++ LibTorch 加载

### 4.2 图像预处理与识别

#### 4.2.1 预处理流程

用户在画板上写的数字，不能直接送给模型——需要先转换成和 MNIST 训练数据相同的格式。这个预处理过程在 C++ 端的 `DigitRecognizer::preprocess()` 中实现，分为五个步骤：

**第一步：转为灰度图。** 如果输入是彩色图片（3 通道或 4 通道），先转为单通道灰度图，去掉颜色信息的干扰。

**第二步：找到笔迹边界。** 逐像素扫描整张图片，找出灰度值低于 245 的像素（即"墨迹"区域）的上下左右边界。这一步确定了用户写的数字在画布上的实际位置。

**第三步：居中裁切为正方形。** 以笔迹边界为基础，四周各扩展 4 个像素的边距，然后裁切为正方形（以宽高中较长的一边为准），短边方向用白色填充。这样做是为了让数字居中，并且保持原始的宽高比。

**第四步：缩放到 28×28。** 使用 OpenCV 的 `cv::resize` 函数将正方形图片缩放到 28×28 像素。缩放使用 `INTER_AREA` 算法，它在缩小图片时效果比较好，不会产生明显的锯齿。

**第五步：归一化。** 将像素值从 0-255 映射到 0-1 范围，然后做反转（因为画板是白底黑字，而 MNIST 是黑底白字），最后减去均值 0.1307 并除以标准差 0.3081，和训练时的处理保持一致。

这五步预处理的顺序和参数都是精心设计的，尤其"居中裁切"这一步对识别准确率影响很大——如果用户把数字写在画布角落，不居中就直接缩放的话，模型可能无法正确识别。

以下是 `src/recognizer.cpp` 中预处理的核心实现（已简化部分边界检查）：

```cpp
std::vector<float> DigitRecognizer::preprocess(const cv::Mat& img)
{
    // 第一步：转为灰度图
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = img.clone();

    // 第二步：找到笔迹边界（灰度值 < 245 的像素）
    int left = cols, top = rows, right = -1, bottom = -1;
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col) {
            if (source[row * cols + col] < 245) {
                left  = std::min(left, col);
                top   = std::min(top, row);
                right = std::max(right, col);
                bottom = std::max(bottom, row);
            }
        }

    // 第三步：扩展边距，居中裁切为正方形
    left = std::max(0, left - 4);
    top  = std::max(0, top - 4);
    right  = std::min(cols - 1, right + 4);
    bottom = std::min(rows - 1, bottom + 4);
    int side = std::max(right - left + 1, bottom - top + 1);
    cv::Mat padded(side, side, CV_8UC1);
    std::fill(padded.data, padded.data + side * side, 255);  // 白色填充

    // 将笔迹复制到正方形中心
    int offsetX = (side - (right - left + 1)) / 2;
    int offsetY = (side - (bottom - top + 1)) / 2;
    // ... memcpy 到 padded 对应位置 ...

    // 第四步：缩放到 28×28
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(28, 28), 0.0, 0.0, cv::INTER_AREA);

    // 第五步：归一化（反转 + 减均值除标准差）
    std::vector<float> normalized(28 * 28, 0.0f);
    for (int row = 0; row < 28; ++row)
        for (int col = 0; col < 28; ++col) {
            float ink = 1.0f - float(resizedData[row * 28 + col]) / 255.0f;
            normalized[row * 28 + col] = (ink - 0.1307f) / 0.3081f;
        }
    return normalized;
}
```

#### 4.2.2 模型推理

预处理完成后，将 28×28 的像素数据转换为形状为 `[1, 1, 28, 28]` 的张量（batch 维度为 1，通道数为 1，高和宽各 28），送入 TorchScript 模型。模型输出一个长度为 10 的向量（logits），取其中最大值的索引作为识别结果（0-9 中的一个数字）。

推理过程使用 `torch::InferenceMode` 上下文管理器，它会关闭梯度计算和一些训练时才需要的簿记操作，让推理速度更快、内存占用更少。

以下是 `src/recognizer.cpp` 中的推理代码：

```cpp
int DigitRecognizer::predict(const cv::Mat& inputImage)
{
    std::vector<float> processed = preprocess(inputImage);
    if (processed.size() != 28 * 28)
        throw std::runtime_error("Input image is empty");

    // 将预处理后的数据转为 [1, 1, 28, 28] 的张量
    auto input = torch::from_blob(
        processed.data(), {1, 1, 28, 28},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    input = input.to(device);   // 送到 CPU 或 GPU

    torch::InferenceMode guard;  // 关闭梯度计算
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input);

    // 前向传播，取概率最大的类别
    torch::Tensor logits = model.forward(inputs).toTensor().to(torch::kCPU);
    return static_cast<int>(logits.argmax(1).item<int64_t>());
}
```

#### 4.2.3 设备选择与预热

C++ 推理端支持 CPU 和 CUDA 两种推理设备，通过界面复选框控制：

- 程序启动时默认使用 CPU 推理
- 如果系统检测到 CUDA 可用（存在 NVIDIA GPU + CUDA 驱动 + cuDNN），"使用 CUDA 进行推理"复选框会被启用，用户勾选后切换为 CUDA 推理
- 如果 CUDA 不可用，复选框被禁用并显示提示，确保推理设备始终为 CPU
- 切换设备时，程序会销毁当前识别器并以新设备重新加载模型

当推理设备为 GPU 时，程序会在模型加载完成后执行一次"预热"：用一个全零的虚拟输入跑一次前向传播，并调用 `torch::cuda::synchronize()` 等待完成。这是因为 CUDA 的第一次推理会触发 kernel 编译和显存分配，如果不提前预热，用户第一次点击"识别"按钮时会明显感到卡顿。

CPU 推理不需要预热，因为 CPU 上没有 kernel 编译的问题。

以下是 `src/mainwindow.cpp` 中 CUDA 切换和模型加载的核心代码：

```cpp
void MainWindow::onCudaToggled(bool checked)
{
    if (checked && !cudaAvailable_) {
        appendLog("CUDA 不可用，无法切换到 CUDA 推理。");
        if (cudaCheckBox_ != nullptr) {
            cudaCheckBox_->blockSignals(true);
            cudaCheckBox_->setChecked(false);
            cudaCheckBox_->blockSignals(false);
        }
        return;
    }
    appendLog(QString("推理设备切换: %1").arg(checked ? "CUDA" : "CPU"));
    loadRecognizer();
}

void MainWindow::loadRecognizer()
{
    const bool useCuda = cudaCheckBox_ != nullptr && cudaCheckBox_->isChecked();
    const QString modelPath = resolveModelPath();
    // ...
    recognizer_ = new DigitRecognizer(modelPath.toStdString(), useCuda);
    recognizer_->warmUp();
    // ...
}
```

以下是 `src/recognizer.cpp` 中的设备选择和预热代码：

```cpp
// 构造函数：根据 useCuda 参数决定设备
DigitRecognizer::DigitRecognizer(const std::string& modelPath, bool useCuda)
{
    if (useCuda) {
        if (!torch::cuda::is_available())
            throw std::runtime_error("CUDA requested but not available.");
        device = torch::Device(torch::kCUDA);
        deviceName_ = "cuda";
    } else {
        device = torch::Device(torch::kCPU);
        deviceName_ = "cpu";
    }
    model = torch::jit::load(modelPath, device);
    model.eval();
}

// CUDA 预热：做一次空推理，触发 kernel 编译
void DigitRecognizer::warmUp()
{
    if (device.type() != torch::kCUDA) return;  // CPU 不需要预热

    torch::InferenceMode guard;
    auto warmInput = torch::zeros(
        {1, 1, 28, 28},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(warmInput);
    (void)model.forward(inputs).toTensor();
    torch::cuda::synchronize();  // 等待 GPU 计算完成
}
```

### 4.3 Qt 界面与画布

#### 4.3.1 整体布局

主窗口采用左右分栏布局：

**左侧** 分为上下两部分：上方是"隔空书写预览"区域，包含摄像头选择下拉框、摄像头预览画面（400×240 像素的深色区域）和"开启隔空书写"按钮；下方是"手写画布"区域，提供 280×280 像素的白色画布供用户鼠标书写。

**右侧** 是控制面板，从上到下依次为：识别结果标签、可信度标签、隔空手势说明、摄像头镜像开关、CUDA 推理开关、操作提示、"识别"按钮、"清空"按钮、操作日志区域。

整个界面使用了现代的卡片式设计风格，控件有圆角和阴影效果，按钮有 hover 状态的颜色变化，整体视觉效果比较整洁。

#### 4.3.2 画布控件

画布 `Canvas` 继承自 `QWidget`，内部维护一个 `QPixmap` 作为底图。绘图逻辑如下：

- 鼠标按下时记录起始点并画一个点
- 鼠标移动时在上一个点和当前点之间画一条线段
- 鼠标抬起时结束当前笔画

画笔设置为黑色、20 像素粗、圆头、抗锯齿，这样画出来的线条比较自然，接近真实书写的感觉。

画布还支持动态调整大小——当窗口尺寸变化时，画布会自动缩放并保持已有的笔迹内容。

以下是 `src/canvas.cpp` 中画布的核心绘图代码：

```cpp
// 画一条线段（鼠标按下后移动时调用）
void Canvas::drawLineTo(const QPoint& endPoint)
{
    QPainter painter(&pixmap_);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(Qt::black, 20, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(lastPoint_, endPoint);

    const int radius = 24;
    update(QRect(lastPoint_, endPoint).normalized()
           .adjusted(-radius, -radius, radius, radius));
    lastPoint_ = endPoint;
}

// 鼠标事件驱动绘制
void Canvas::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton)
        beginStroke(event->pos());
}
void Canvas::mouseMoveEvent(QMouseEvent* event) {
    if ((event->buttons() & Qt::LeftButton) && drawing_)
        appendStroke(event->pos());
}
void Canvas::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && drawing_) {
        appendStroke(event->pos());
        endStroke();
    }
}
```

#### 4.3.3 推理设备切换

界面右侧提供了"使用 CUDA 进行推理"复选框。程序启动时会检测 CUDA 可用性：如果系统有 NVIDIA GPU 且 libtorch 的 CUDA 支持正常，复选框会被启用；否则复选框被禁用并提示"未检测到可用的 CUDA 设备"。

默认状态下复选框未勾选，使用 CPU 推理。用户勾选后，程序会销毁当前的识别器实例，以 CUDA 设备重新加载模型文件，并执行预热。取消勾选则切换回 CPU 推理。整个过程在日志区域有详细记录。

项目只使用一份模型文件 `mnist_model.pt`，模型文件本身不绑定设备，CPU 和 CUDA 都加载同一份文件。

模型文件的查找支持回退机制：搜索路径包括可执行文件目录、上级目录、`artifacts/models`、`dist/models` 等多个位置，确保在开发环境和发布环境中都能正常工作。

以下是 `src/mainwindow.cpp` 中设备切换与模型加载的核心代码：

```cpp
void MainWindow::onCudaToggled(bool checked)
{
    if (checked && !cudaAvailable_) {
        appendLog("CUDA 不可用，无法切换到 CUDA 推理。");
        cudaCheckBox_->blockSignals(true);
        cudaCheckBox_->setChecked(false);
        cudaCheckBox_->blockSignals(false);
        return;
    }
    appendLog(QString("推理设备切换: %1").arg(checked ? "CUDA" : "CPU"));
    loadRecognizer();
}

void MainWindow::loadRecognizer()
{
    const bool useCuda = cudaCheckBox_ != nullptr && cudaCheckBox_->isChecked();
    const QString modelPath = resolveModelPath();
    appendLog(QString("准备加载模型: useCuda=%1, path=%2").arg(useCuda, modelPath));

    if (modelPath.isEmpty()) {
        setRecognitionEnabled(false);
        resultLabel_->setText("识别结果：模型加载失败。");
        return;
    }

    setRecognitionBusy(true);
    delete recognizer_;
    recognizer_ = nullptr;

    try {
        recognizer_ = new DigitRecognizer(modelPath.toStdString(), useCuda);
        recognizer_->warmUp();
        setRecognitionEnabled(true);
        appendLog(QString("模型加载成功: device=%1")
                  .arg(QString::fromStdString(recognizer_->deviceName())));
    } catch (const std::exception& error) {
        recognizer_ = nullptr;
        setRecognitionEnabled(false);
        appendLog(QString("模型加载失败: %1").arg(error.what()));
    }
    setRecognitionBusy(false);
}
```

模型文件查找逻辑在多个候选路径中搜索 `mnist_model.pt`：

```cpp
QString resolveModelPathWithFallback()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QStringList roots = {
        QDir(appDir).filePath("models"),
        QDir(appDir).filePath("../models"),
        QDir(appDir).filePath("../../../artifacts/models"),
        QDir(appDir).filePath("../dist/models"),
        // ...
    };
    for (const QString& root : roots) {
        QString candidate = candidateModelPath(root);
        if (!candidate.isEmpty()) return candidate;
    }
    return {};
}
```

#### 4.3.4 操作日志

主窗口右侧有一个操作日志区域，会实时记录程序运行中的关键事件：模型加载状态、识别请求和结果、摄像头启停、隔空书写启停、异常信息等。每条日志都带有时间戳，方便调试和问题排查。

### 4.4 隔空书写

隔空书写是整个项目中最复杂、工程性最强的部分。它涉及摄像头采集、手部关键点检测、坐标映射、轨迹平滑、手势判定等多个环节，需要 Qt（C++）和 Python 两种语言协作完成。

#### 4.4.1 架构设计

整个隔空书写的数据流如下：

1. Qt 端通过 `QCamera` 和 `QVideoSink` 采集摄像头画面并实时显示
2. Qt 端将画面压缩为 JPEG 格式，通过 `QProcess` 的 stdin 以长度前缀协议（4 字节大端长度 + JPEG 数据）发送给 Python 进程
3. Python 进程（`hand_tracker_service.py`）接收帧数据，使用 MediaPipe 的 HandLandmarker 模型检测手部 21 个关键点
4. Python 进程通过 stdout 输出轻量的 JSON 数据（包含指尖坐标、是否有手、是否在书写等状态）
5. Qt 端解析 JSON 数据，将指尖坐标映射到画布坐标，驱动画笔绘制

这个架构的关键设计决策是：**图像数据只从 Qt 流向 Python（单向），Python 只返回坐标数据（不返回图像）。** 之前的旧方案中，Python 会把带有标记的画面回传给 Qt 显示，这导致管道阻塞和严重延迟。现在改为 Qt 自己负责画面显示（直接用原始摄像头画面），Python 只做"计算"不做"渲染"，性能问题从根本上被解决了。

以下是 `src/airwritecontroller.cpp` 中 Qt 向 Python 发送 JPEG 帧的核心代码，采用长度前缀协议避免粘包：

```cpp
void AirWriteController::trySendPendingFrame()
{
    // 压缩为 JPEG（降低分辨率 + 降低质量，减小传输量）
    QImage frame = pendingFrame_;
    if (frameMaxWidth_ > 0 && frame.width() > frameMaxWidth_) {
        double scale = double(frameMaxWidth_) / double(frame.width());
        frame = frame.scaled(frameMaxWidth_, int(frame.height() * scale),
                             Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }
    QByteArray jpegBytes;
    QBuffer buffer(&jpegBytes);
    frame.save(&buffer, "JPG", frameJpegQuality_);  // 质量 70

    // 长度前缀协议：4 字节大端长度 + JPEG 数据
    QByteArray packet;
    char header[4];
    qToBigEndian<quint32>(jpegBytes.size(), reinterpret_cast<uchar*>(header));
    packet.append(header, 4);
    packet.append(jpegBytes);

    trackerProcess_.write(packet);  // 通过 stdin 发送给 Python
}
```

Python 端（`hand_tracker_service.py`）接收帧后只返回轻量 JSON：

```python
# stdout 输出协议示例
payload = {
    "type": "frame",
    "has_hand": last_has_hand,
    "drawing_active": last_drawing_active,
    "frame_size": [frame.shape[1], frame.shape[0]],
    "cursor": last_cursor,   # [x, y] 食指指尖像素坐标
}
emit(payload)  # emit() 将 dict 序列化为一行 JSON 写入 stdout
```

#### 4.4.2 手部检测

Python 端使用 Google 的 MediaPipe Tasks 库进行手部关键点检测。MediaPipe HandLandmarker 可以在一张图片中检测到一只手的 21 个关键点，包括指尖、指关节、手腕等位置。

检测过程在后台线程中运行，不会阻塞主循环。当 Qt 端发送新帧时，后台线程取最新的一帧进行处理；如果处理速度跟不上帧率，旧帧会被丢弃，只保留最新结果。这种"取最新、丢旧帧"的策略保证了追踪的实时性。

以下是 `hand_tracker_service.py` 中 MediaPipe 模型的初始化和检测线程代码：

```python
# 初始化 HandLandmarker（首次运行时自动下载模型文件）
model_path = ensure_model_available()
options = mp_tasks.vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
    running_mode=mp_tasks.vision.RunningMode.IMAGE,
    num_hands=1,                           # 只检测一只手
    min_hand_detection_confidence=0.72,
    min_hand_presence_confidence=0.68,
    min_tracking_confidence=0.72,
)
landmarker = mp_tasks.vision.HandLandmarker.create_from_options(options)
```

```python
# 后台检测线程：只取最新帧，丢弃旧帧
def detection_worker():
    while not stop_event.is_set():
        frame_copy = None
        with detection_lock:
            if latest_detection_frame["frame"] is not None:
                frame_copy = latest_detection_frame["frame"]
                latest_detection_frame["frame"] = None   # 标记已消费

        if frame_copy is None:
            time.sleep(0.005)
            continue

        mp_img = Image(image_format=ImageFormat.SRGB,
                       data=np.ascontiguousarray(frame_copy))
        results = landmarker.detect(mp_img)

        with detection_lock:
            detection_results["hand_landmarks"] = (
                results.hand_landmarks if results.hand_landmarks else None
            )
```

#### 4.4.3 手势判定

系统使用"食指抬起"作为书写的触发手势。具体判定逻辑如下：

**手指伸展检测：** 对于每根手指（食指、中指、无名指、小指），通过指尖（TIP）、中间关节（PIP）和根部关节（MCP）的相对位置关系来判断手指是否伸展。计算方法考虑了指尖到手掌中心的距离、手指骨骼的投影长度等因素。

**书写置信度：** 不是简单的"伸/不伸"二值判断，而是一个 0 到 1 的连续置信度分数。它综合考虑了食指伸展程度、中指/无名指/小指的弯曲程度、食指指尖锁定的信任度等因素。

**状态机：** 书写状态的切换不是瞬间完成的，而是需要连续若干帧的确认。从"跟踪"切换到"书写"需要 2 帧确认，从"书写"切换回"跟踪"需要 6 帧确认。这种滞后机制避免了单帧误检导致的频繁切换。

以下是 `hand_tracker_service.py` 中手指伸展评分和书写置信度的核心代码：

```python
def finger_extension_score(landmarks, tip_idx, pip_idx, mcp_idx,
                           palm_center, palm_span):
    """判断单根手指是否伸展：指尖到掌心距离 vs 关节到掌心距离"""
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]

    tip_to_center = math.hypot(tip.x - palm_center[0],
                               tip.y - palm_center[1]) / palm_span
    pip_to_center = math.hypot(pip.x - palm_center[0],
                               pip.y - palm_center[1]) / palm_span

    # 手指骨骼投影：指尖沿骨骼方向的伸展程度
    bone_x = pip.x - mcp.x
    bone_y = pip.y - mcp.y
    tip_x  = tip.x - mcp.x
    tip_y  = tip.y - mcp.y
    bone_len = math.hypot(bone_x, bone_y)
    proj = (tip_x * bone_x + tip_y * bone_y) / bone_len if bone_len > 1e-6 else 0

    return (tip_to_center - pip_to_center) + (proj / max(1e-6, bone_len) - 1.0) * 0.35
```

```python
def compute_writing_confidence(landmarks, hand_scale_px, index_trusted):
    """综合食指伸展、其他手指弯曲、指尖锁定信任度，输出 0~1 置信度"""
    # ... 计算各手指 extension_score ...
    curl_penalty = (max(0.0, middle_score - 0.10)
                  + max(0.0, ring_score - 0.08)
                  + max(0.0, little_score - 0.06))
    tip_to_palm = math.hypot(landmarks[INDEX_TIP].x - palm_center[0],
                             landmarks[INDEX_TIP].y - palm_center[1]) / palm_span
    confidence = (max(0.0, index_score - 0.12) * 1.15
                + max(0.0, tip_to_palm - 0.55) * 0.55
                - curl_penalty * 0.30)
    if index_trusted:
        confidence += 0.22
    return max(0.0, min(1.0, confidence))
```

```python
# 状态机：书写状态切换需要多帧确认
def update_drawing_state(self, raw_active, confidence):
    if raw_active and confidence >= 0.38:
        self.draw_confirm_frames += 1
        self.idle_confirm_frames = 0
        if not self.drawing_active and self.draw_confirm_frames >= 2:
            self.drawing_active = True    # 连续 2 帧确认 → 开始书写
    else:
        self.idle_confirm_frames += 1
        if self.drawing_active and self.idle_confirm_frames >= 6 and confidence < 0.25:
            self.drawing_active = False   # 连续 6 帧确认 → 停止书写
    return self.drawing_active
```

#### 4.4.4 食指锁定与轨迹稳定

MediaPipe 的手部检测并不总是完美的——有时食指的检测结果会突然跳到其他手指的位置，或者在手指交叉时产生误检。为了解决这个问题，系统设计了"食指锁定"机制：

当检测到的食指位置与历史轨迹的预测位置偏差较大时，系统不会立刻跟随新的检测结果，而是短时间（最多 4 帧）沿历史速度方向继续预测，等待检测结果稳定后再重新接纳。同时，系统还对指尖位置做限幅平滑——每一帧的移动距离不超过一个基于手部大小动态计算的上限，避免出现"瞬移"现象。

对于短时的手部丢失（1-2 帧），系统会保持上一个已知位置并沿速度方向预测，而不是立刻停止书写，减少画笔意外中断的情况。

以下是 `hand_tracker_service.py` 中 `TipLockState` 的核心实现：

```python
@dataclass
class TipLockState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0       # 速度分量
    vy: float = 0.0
    valid: bool = False
    inconsistent_frames: int = 0  # 连续不一致帧数

    def update(self, raw_x, raw_y, hand_scale, trusted):
        if not self.valid:
            self.x, self.y = raw_x, raw_y
            self.valid = True
            return self.x, self.y

        # 计算预测位置与实际检测的偏差
        predicted_x = self.x + self.vx
        predicted_y = self.y + self.vy
        mismatch = math.hypot(raw_x - predicted_x, raw_y - predicted_y)
        mismatch_limit = max(20.0, hand_scale * 0.65)

        if trusted and mismatch <= mismatch_limit:
            # 检测结果可信，直接采纳
            target_x, target_y = raw_x, raw_y
            self.inconsistent_frames = 0
        else:
            # 检测结果不可信，沿历史轨迹预测（最多 4 帧）
            self.inconsistent_frames += 1
            if self.inconsistent_frames <= 4:
                target_x, target_y = predicted_x, predicted_y
            else:
                blend = 0.20  # 4 帧后缓慢重新接纳检测结果
                target_x = predicted_x * 0.80 + raw_x * 0.20
                target_y = predicted_y * 0.80 + raw_y * 0.20

        # 限幅：单帧移动距离不超过 hand_scale * 0.50
        max_step = max(16.0, hand_scale * 0.50)
        dx, dy = target_x - self.x, target_y - self.y
        distance = math.hypot(dx, dy)
        if distance > max_step:
            scale = max_step / distance
            target_x = self.x + dx * scale
            target_y = self.y + dy * scale

        # 更新速度（指数移动平均）
        new_vx = target_x - self.x
        new_vy = target_y - self.y
        self.vx = self.vx * 0.45 + new_vx * 0.55
        self.vy = self.vy * 0.45 + new_vy * 0.55
        self.x, self.y = target_x, target_y
        return self.x, self.y
```

#### 4.4.5 Qt 端坐标映射与绘制

Qt 端收到 Python 返回的指尖坐标后，需要将其映射到画布坐标。映射公式是简单的线性缩放：

画布 X = 摄像头坐标 X × (画布宽度 / 摄像头帧宽度)
画布 Y = 摄像头坐标 Y × (画布高度 / 摄像头帧高度)

映射后的坐标还会经过一轮平滑处理（指数移动平均，系数根据移动距离动态调整），进一步减少抖动。

绘制时，如果两帧之间的距离超过 150 像素，系统会判定为"跳变"并自动断笔，避免出现跨屏幕的连线。如果距离在 6 到 150 像素之间，系统会在两个点之间做插值，保证线条的连续性。

停笔也有缓冲设计：当 Python 端报告"未在书写"时，Qt 端不会立刻停笔，而是等待 120 毫秒再确认，避免一帧的误检就把笔画切断。

以下是 `src/mainwindow.cpp` 中坐标映射和隔空绘制的核心代码：

```cpp
// 坐标映射：摄像头坐标 → 画布坐标
QPointF MainWindow::mapCameraToCanvas(
    const QPointF& cameraPos, const QSize& cameraFrameSize, const QSize& canvasSize)
{
    double xRatio = double(canvasSize.width())  / double(cameraFrameSize.width());
    double yRatio = double(canvasSize.height()) / double(cameraFrameSize.height());
    double mappedX = cameraPos.x() * xRatio;
    double mappedY = cameraPos.y() * yRatio;
    // 限制在画布范围内
    return QPointF(
        std::clamp(mappedX, 0.0, double(canvasSize.width() - 1)),
        std::clamp(mappedY, 0.0, double(canvasSize.height() - 1))
    );
}
```

```cpp
// 接收追踪数据后的平滑与绘制逻辑
void MainWindow::onAirTrackingUpdated(
    const QPointF& cursorPoint, const QSize& frameSize, bool drawingActive)
{
    QPointF canvasPoint = mapCameraToCanvas(cursorPoint, frameSize, canvas_->size());

    // 指数移动平均平滑（系数根据移动距离动态调整）
    double movement = std::hypot(canvasPoint.x() - airSmoothedPoint_.x(),
                                 canvasPoint.y() - airSmoothedPoint_.y());
    double alpha = std::clamp(0.34 + movement / 220.0, 0.34, 0.68);
    airSmoothedPoint_ = QPointF(
        airSmoothedPoint_.x() * (1.0 - alpha) + canvasPoint.x() * alpha,
        airSmoothedPoint_.y() * (1.0 - alpha) + canvasPoint.y() * alpha);

    if (drawingActive) {
        airStrokeReleasePending_ = false;
        if (!airStrokeActive_) {
            canvas_->beginStroke(airSmoothedPoint_.toPoint());
            airStrokeActive_ = true;
        } else {
            double drawDistance = std::hypot(/* delta */);
            if (drawDistance > 150.0) {
                finalizeAirStroke();    // 跳变 → 自动断笔
            } else if (drawDistance > 6.0) {
                // 插值绘制，保证线条连续
                int steps = int(std::min(12.0, std::max(2.0, drawDistance / 6.0)));
                for (int s = 1; s <= steps; ++s) {
                    QPoint interp = /* 线性插值 */;
                    canvas_->appendStroke(interp);
                }
            }
        }
    } else if (airStrokeActive_) {
        // 停笔缓冲：等待 120ms 确认
        if (!airStrokeReleasePending_) {
            airStrokeReleasePending_ = true;
            airStrokeReleaseTimer_.restart();
        } else if (airStrokeReleaseTimer_.elapsed() >= 120) {
            finalizeAirStroke();
        }
    }
}
```

### 4.5 构建系统与工程化部署

本项目同时维护两套构建入口，分别面向不同的使用场景：

**命令行 CMake 构建**用于可复现的发布流程。构建命令通过 `-DCMAKE_PREFIX_PATH` 和 `-DTorch_DIR` 显式指定 Qt 和 LibTorch 的路径，使用 Visual Studio 2022 作为生成器。CMake 的 `find_package(Torch)` 会自动检测 CUDA 支持，配置正确的编译定义和链接器选项，包括关键的 `-INCLUDE:?warp_size@cuda@at@@YAHXZ` 链接器强制包含符号。构建完成后，`scripts/package_release.ps1` 负责将 Qt 运行库、LibTorch DLL 和模型文件复制到 `dist/` 发布目录，并生成一键启动脚本 `run_handwriting_recog.bat`。

**Qt Creator qmake 构建**用于日常开发调试。`HandwritingRecognition.pro` 中手动配置了 LibTorch 的头文件路径和库链接，并通过 `exists()` 条件判断自动检测 CUDA 版 LibTorch 并链接相应库。构建完成后，`QMAKE_POST_LINK` 自动调用 `scripts/deploy_qt_creator_build.ps1` 将 DLL 和模型文件部署到构建目录。

两种构建方式编译的是同一套源码，但在链接器配置上存在差异。CMake 通过 `find_package(Torch)` 自动注入的链接器选项，需要在 qmake 中手动添加，否则 MSVC 链接器的死代码消除会移除 CUDA 初始化入口，导致运行时无法检测到 CUDA。

项目还处理了 LibTorch 与 Qt 的宏冲突问题：两者都定义了 `slots` 和 `signals` 宏，直接包含 LibTorch 头文件会导致编译错误。`src/recognizer.h` 通过在包含 LibTorch 头文件前先 `#undef` 这两个宏、包含后再恢复的方式解决。

---

## 五、实验结果

### 5.1 模型训练结果

在 NVIDIA GPU 上使用 CUDA 进行 50 个 epoch 的训练后，模型在 MNIST 测试集上的准确率约为 99.7%。训练使用了小型 CNN、数据增强、混合精度训练（AMP）、TF32、OneCycleLR 学习率调度和标签平滑等策略。

训练产物：
- `model.pth`：PyTorch 原生权重文件
- `mnist_model.pt`：TorchScript 格式，用于 C++ 推理

### 5.2 推理性能

单次推理（包含图像预处理和模型前向传播）的耗时：
- CPU 推理：约 1-5 毫秒
- GPU 推理（预热后）：约 1-3 毫秒

对于 MNIST 这种小模型（约 20 万个参数），CPU 和 GPU 的推理速度差异不大，因为模型本身的计算量很小，数据传输的开销可能比计算本身还大。

### 5.3 发布产物

发布包位于 `dist/` 目录，包含：
- `handwriting_recog.exe`：主程序
- Qt 运行库（Qt6Core.dll、Qt6Gui.dll、Qt6Widgets.dll 等）
- LibTorch 运行库（torch_cpu.dll、torch.dll、c10.dll 等，CUDA 版还包含 torch_cuda.dll、c10_cuda.dll）
- `models/cpu/mnist_model.pt` 和 `models/gpu/mnist_model.pt`：TorchScript 模型文件
- `run_handwriting_recog.bat`：一键启动脚本（自动设置 PATH 包含 Qt 和 LibTorch 路径）

---

## 六、遇到的问题与解决方案

**问题一：图像预处理不当导致识别错误。** 早期版本没有做"居中裁切"处理，用户把数字写在画布边缘时，缩放到 28×28 后数字会变形。解决方案是在缩放前先找到笔迹的边界框，将其居中裁切为正方形后再缩放。

**问题二：CUDA 推理首次点击闪退。** 原因是 CUDA 的第一次推理触发 kernel 编译时耗时较长，用户以为程序卡死了就重复点击，导致重入异常。解决方案是加入预热机制（模型加载后立刻做一次空推理）和识别期间的按钮锁定保护。

**问题三：隔空书写严重卡顿。** 旧方案中 Python 会把带有手部标记的画面回传给 Qt 显示，图像在两个进程之间传输导致管道阻塞。解决方案是改为 Qt 自己用 `QCamera` 显示原始画面，Python 只返回轻量的坐标 JSON，不传输任何图像数据。

**问题四：食指追踪不稳定。** MediaPipe 的检测结果偶尔会出现跳变，食指位置突然"跳"到其他手指。解决方案是设计了食指锁定状态机和限幅平滑机制，在检测结果不稳定时沿历史轨迹预测，等检测稳定后再重新接纳。

**问题五：Windows 下多进程数据加载器不稳定。** 使用较多的 `num_workers` 时偶尔出现内存分配失败。原因是 Windows 使用 spawn 方式创建子进程，开销比 Linux 的 fork 大。解决方案是将 Windows 下的 num_workers 限制为不超过 2。

**问题六：LibTorch 和 Qt 的宏冲突。** LibTorch 的头文件和 Qt 的信号/槽机制都定义了 `slots` 和 `signals` 宏，直接包含会导致编译错误。解决方案是在包含 LibTorch 头文件之前先取消这两个宏的定义，包含之后再恢复。

**问题七：Qt Creator 构建后 CUDA 不可用。** 命令行 CMake 构建的程序能正常检测 CUDA，但 Qt Creator 的 qmake 构建却显示 `CUDA available: false`。经排查发现，CMake 的 `find_package(Torch)` 会自动向 MSVC 链接器注入 `-INCLUDE:?warp_size@cuda@at@@YAHXZ` 选项，强制包含 LibTorch CUDA 后端的初始化符号。qmake 构建缺少该选项时，链接器的死代码消除会连带移除 CUDA 初始化入口，导致运行时无法检测到 CUDA。解决方案是在 `HandwritingRecognition.pro` 的 CUDA 链接块中添加 `QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ`。

---

## 七、总结与展望

本项目完成了一个完整的手写数字识别桌面应用，从模型训练到部署使用形成了闭环。项目不仅实现了基本的画板书写和识别功能，还扩展了隔空书写这一创新交互方式，涉及深度学习、桌面应用开发、计算机视觉、多进程通信等多个技术领域的综合运用。

通过这个项目，我体会到模型训练只是整个开发过程的一小部分。如何把训练好的模型集成到桌面程序中稳定运行、如何处理图像预处理中的各种边界情况、如何让多个进程高效协作而不阻塞，这些工程问题往往比调模型参数更花时间，也更贴近实际的软件开发场景。

可能的改进方向包括：
- 支持多位数字的连续书写和识别
- 使用更大规模的数据集或预训练模型提升泛化能力
- 将 MediaPipe 手部检测移植到 C++ 端，消除对 Python 环境的依赖
- 添加用户自定义训练功能，允许用户用自己的手写样本微调模型

---

## 参考文献

[1] 淼学派对. 使用Python实现隔空手势控制鼠标[EB/OL]. 腾讯云, 2023. https://cloud.tencent.com/developer/article/2340720.

[2] 嗔笑. 基于PyTorch搭建简单网络实现MNIST手写数字识别[EB/OL]. 知乎, 2025. https://zhuanlan.zhihu.com/p/693122282.

[3] jessicalh. Qt6 Skills[CP/OL]. GitHub, 2026. https://github.com/jessicalh/qt-skill-claude.

[4] ding-juncai. AI-Virtual-Painter[CP/OL]. Gitee, 2025. https://gitee.com/ding-juncai/ai-virtual-painter.

[5] Qt Group. Qt开发文档[EB/OL]. https://doc.qt.ac.cn/.
