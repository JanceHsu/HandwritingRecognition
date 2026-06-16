# Python 脚本说明

本项目共有三个 Python 脚本，均位于 `scripts/` 目录下。它们分别负责模型训练、手部追踪服务和测试图片导出。三个脚本相互独立，各自通过命令行参数驱动。

---

## 1. train_mnist.py —— 模型训练与导出

**路径**：`scripts/train_mnist.py`

**用途**：在 MNIST 数据集上训练一个小型卷积神经网络，并将训练好的模型导出为 TorchScript 格式，供 C++ LibTorch 加载推理。

### 1.1 网络结构

脚本定义了 `NeuralNetwork` 类，继承自 `nn.Module`，是一个由卷积层和全连接层组成的小型 CNN。

**特征提取部分（`self.features`）**：

```
Conv2d(1, 32, 3x3, padding=1) → BatchNorm2d(32) → ReLU
Conv2d(32, 32, 3x3, padding=1) → BatchNorm2d(32) → ReLU
MaxPool2d(2) → Dropout2d(0.10)

Conv2d(32, 64, 3x3, padding=1) → BatchNorm2d(64) → ReLU
Conv2d(64, 64, 3x3, padding=1) → BatchNorm2d(64) → ReLU
MaxPool2d(2) → Dropout2d(0.20)
```

输入图像为 1x28x28（单通道灰度），经过两组卷积+批归一化+ReLU 后，两次 MaxPool2d(2) 将特征图从 28x28 缩小到 14x14 再到 7x7。两层 Dropout2d 随机丢弃整个卷积通道，防止过拟合。

**分类部分（`self.classifier`）**：

```
Flatten
Linear(64 * 7 * 7, 256) → ReLU → Dropout(0.3)
Linear(256, 128) → ReLU → Dropout(0.2)
Linear(128, 10)
```

将 64x7x7=3136 维的特征向量逐步映射到 10 个类别（数字 0-9）。Dropout 层在训练时随机丢弃神经元，推理时自动关闭。

`forward()` 方法直接将输入依次通过 `self.features` 和 `self.classifier`，返回 logits 张量。

### 1.2 数据加载与增强

`build_dataloaders()` 函数负责构建训练和测试数据加载器。

**训练集增强**：

| 增强方式 | 参数 | 作用 |
|---------|------|------|
| RandomRotation | +-15 度 | 模拟手写时数字的倾斜 |
| RandomAffine | +-10 度旋转、10% 平移、0.9-1.1 缩放、+-8 度错切 | 模拟位置偏移、大小变化和纸张变形 |
| RandomPerspective | 15% 畸变程度，25% 概率触发 | 模拟透视变形（如纸张弯曲） |
| ColorJitter | 15% 亮度和对比度变化，50% 概率触发 | 模拟不同光照条件 |
| ToTensor | — | 将 PIL 图像转为 [0, 1] 范围的张量 |
| Normalize | mean=0.1307, std=0.3081 | MNIST 数据集的统计量，使输入分布更规整 |

**测试集**：只做 ToTensor 和 Normalize，不做增强，以保证评估结果的可比性。

MNIST 数据集默认从 `https://storage.googleapis.com/cvdf-datasets/mnist/` 下载，可通过 `--mirror-url` 参数切换镜像源。

### 1.3 训练策略

**优化器**：AdamW，学习率 1e-3，权重衰减 5e-4。AdamW 将权重衰减从梯度更新中分离出来，正则化效果更稳定。

**学习率调度**：OneCycleLR，核心参数如下：
- `pct_start=0.15`：前 15% 的训练步数为预热阶段，学习率从 `max_lr / 10` 逐渐升高到 `max_lr`
- `div_factor=10.0`：初始学习率 = max_lr / 10
- `final_div_factor=100.0`：最终学习率 = max_lr / 1000
- 每个训练批次（batch）更新一次调度器

这种"先升后降"的学习率策略有助于模型在训练初期快速探索参数空间，后期精细调整以收敛到更优解。

**损失函数**：`CrossEntropyLoss(label_smoothing=0.05)`。标签平滑将 5% 的概率分配给错误类别，防止模型对训练标签过于"自信"，减少过拟合。

**随机种子**：`torch.manual_seed(42)` 和 `torch.cuda.manual_seed_all(42)`，保证多次运行的可复现性。

### 1.4 GPU 优化

当 CUDA 可用时，脚本自动启用以下全局优化：

```python
torch.backends.cudnn.benchmark = True       # 自动选择最快卷积算法
torch.set_float32_matmul_precision("high")  # 允许 TF32 精度
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

训练循环中的 GPU 特化：

- **channels_last 内存格式**：`inputs.to(memory_format=torch.channels_last)` 将通道维度放在最后，更适合 GPU 上的 cuDNN 卷积计算
- **混合精度训练（AMP）**：通过 `torch.autocast(device_type="cuda", dtype=torch.float16)` 将前向传播中的矩阵乘法和卷积自动降为 float16，速度约为 float32 的两倍；反向传播仍用 float32 保证精度
- **GradScaler**：先放大 loss 再计算梯度，算完后缩放回去，防止 float16 下梯度太小而被"吞掉"
- **set_to_none=True**：`optimizer.zero_grad(set_to_none=True)` 直接将梯度设为 None 而非填零，减少内存操作

**DataLoader 优化**（仅 GPU 模式）：

| 参数 | 说明 |
|------|------|
| num_workers | 数据加载子进程数，Windows 上限 2（spawn 方式开销大） |
| pin_memory | 启用锁页内存，GPU 通过 DMA 直接读取，速度更快 |
| persistent_workers | 子进程在整个训练过程中保持存活，避免每轮重建 |

`--no-augment` 参数可关闭数据增强，用于快速验证代码是否能正常运行。

### 1.5 训练循环

`train()` 函数执行一个 epoch 的训练：

1. 将数据送到目标设备（CPU 或 CUDA），使用 `non_blocking=True` 异步传输
2. 在 `autocast` 上下文中执行前向传播，得到预测值
3. 计算损失
4. 如果使用 GradScaler：`scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`
5. 否则：`loss.backward()` → `optimizer.step()`
6. 更新学习率调度器
7. 每 100 个批次打印一次当前损失

`test()` 函数在测试集上评估准确率：

1. 切换到 `model.eval()` 模式（关闭 Dropout，BatchNorm 使用全局统计量）
2. 在 `torch.no_grad()` 上下文中禁用梯度计算
3. 累计测试损失和正确预测数
4. 返回准确率（正确数 / 总数）

### 1.6 模型导出

训练完成后，脚本执行两步导出：

**第一步：保存 PyTorch 原生权重**

```python
torch.save(model.state_dict(), "model.pth")
```

`model.pth` 是标准的 PyTorch state_dict 格式，包含所有可学习参数的字典，供 Python 端后续加载或微调。

**第二步：导出 TorchScript 模型**

```python
model = model.to("cpu")                    # 必须移到 CPU
example = torch.rand(1, 1, 28, 28)         # 模拟输入
traced = torch.jit.trace(model.cpu(), example)  # 跟踪前向传播
traced.save("mnist_model.pt")
```

`torch.jit.trace` 通过执行一次前向传播，记录所有张量操作，生成与设备无关的计算图。导出时必须将模型移到 CPU，这样生成的 `.pt` 文件不依赖任何特定设备，在 C++ 端可以加载到 CPU 或 CUDA 上运行。

### 1.7 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root` | `data` | MNIST 数据集存放目录 |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `256` | 批大小（GPU 训练推荐改为 64） |
| `--lr` | `1e-3` | 学习率 |
| `--output-dir` | `artifacts` | 模型输出目录 |
| `--mirror-url` | Google Cloud Storage | MNIST 下载镜像 |
| `--device` | `auto` | 设备选择：cpu / cuda / auto |
| `--num-workers` | `2` | DataLoader 工作进程数 |
| `--pin-memory` | 关闭 | 启用锁页内存 |
| `--no-augment` | 关闭 | 关闭数据增强 |

### 1.8 典型命令

```powershell
# GPU 正式训练
py -3.12 scripts\train_mnist.py --epochs 50 --batch-size 64 --output-dir artifacts\models --device cuda --pin-memory --num-workers 8

# CPU 快速验证
py -3.12 scripts\train_mnist.py --epochs 3 --batch-size 256 --output-dir artifacts\models --device cpu --no-augment --num-workers 0
```

---

## 2. hand_tracker_service.py —— 手部追踪服务

**路径**：`scripts/hand_tracker_service.py`

**用途**：作为隔空书写功能的后端服务，从 Qt 主程序接收摄像头帧，使用 MediaPipe 检测手部关键点，输出食指指尖坐标和书写状态。

### 2.1 架构概述

脚本运行时涉及三个线程的协作：

```
主线程
├─ 采集线程（capture_worker）：从 stdin 读取 JPEG 帧，或直接打开摄像头采集
├─ 检测线程（detection_worker）：消费最新帧，运行 MediaPipe HandLandmarker
└─ 主循环：读取检测结果，计算书写状态，输出 JSON 到 stdout
```

**与 Qt 的通信协议**：

- **输入（stdin）**：Qt 通过 `QProcess` 发送压缩 JPEG 帧，采用长度前缀协议——4 字节大端整数表示 JPEG 数据长度，后跟 JPEG payload
- **输出（stdout）**：每帧输出一行 JSON，格式为 `{"type":"frame","has_hand":true,"drawing_active":true,"frame_size":[640,480],"cursor":[320,240]}`
- **状态消息**：`{"type":"status","level":"info","message":"..."}`

关键设计：**图像数据只从 Qt 流向 Python（单向），Python 只返回坐标数据（不返回图像）**，从根本上避免了旧方案中图像回传导致的管道阻塞问题。

### 2.2 MediaPipe HandLandmarker 初始化

`ensure_model_available()` 函数检查模型文件是否存在，不存在则从 Google Cloud Storage 下载：

```
模型 URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
本地路径: scripts/.cache/hand_landmarker.task
```

下载使用临时文件 + 原子替换（`tmp_path.replace(MODEL_PATH)`），避免下载中断导致文件损坏。

HandLandmarker 配置：
- `running_mode=IMAGE`：逐帧检测模式（非视频流模式）
- `num_hands=1`：只检测一只手
- `min_hand_detection_confidence=0.72`：手部检测最低置信度
- `min_hand_presence_confidence=0.68`：手部存在最低置信度
- `min_tracking_confidence=0.72`：追踪最低置信度

### 2.3 采集线程

脚本支持两种帧来源，通过 `--stdin-frames` 参数切换：

**stdin 模式**（Qt 集成时使用）：

1. 从 `sys.stdin.buffer` 读取 4 字节大端长度头
2. 按长度读取 JPEG payload（上限 8MB）
3. 使用 `cv2.imdecode()` 解码为 BGR 格式的 numpy 数组
4. 可选镜像翻转（`--mirror`）
5. 存入共享变量 `latest_capture_frame`，附带序列号和时间戳

**摄像头模式**（独立调试时使用）：

1. 使用 `cv2.VideoCapture` 打开摄像头（DirectShow 后端）
2. 支持按设备名称（`--camera-name`）或索引（`--camera-index`）选择
3. 设置分辨率和缓冲区大小为 1
4. 循环读取帧，存入共享变量

### 2.4 检测线程

`detection_worker()` 在后台线程中运行 MediaPipe 检测：

1. 从 `latest_detection_frame` 取出最新帧（取最新、丢旧帧策略）
2. 将 BGR 转为 RGB，构造 MediaPipe `Image` 对象
3. 调用 `landmarker.detect(mp_img)` 执行检测
4. 记录检测延迟和处理帧数
5. 将结果存入 `detection_results["hand_landmarks"]`

"取最新丢旧帧"的策略保证了追踪的实时性——如果检测速度跟不上帧率，旧帧会被丢弃，只保留最新结果。

### 2.5 手部关键点与手指状态检测

MediaPipe HandLandmarker 输出 21 个关键点，每个关键点包含归一化的 (x, y, z) 坐标。脚本中定义了关键点索引常量：

| 索引 | 名称 | 位置 |
|------|------|------|
| 0 | WRIST | 手腕 |
| 5 | INDEX_MCP | 食指根部关节 |
| 6 | INDEX_PIP | 食指中间关节 |
| 7 | INDEX_DIP | 食指远端关节 |
| 8 | INDEX_TIP | 食指指尖 |
| 9-12 | MIDDLE_* | 中指各关节 |
| 13-16 | RING_* | 无名指各关节 |
| 17-20 | LITTLE_* | 小指各关节 |

`detect_finger_state()` 函数判断四根手指的伸展状态：

1. 计算掌心中心：手腕和四根手指根部关节（MCP）的平均位置
2. 计算掌心跨度：食指 MCP 到小指 MCP 的距离（用于归一化）
3. 对每根手指调用 `finger_extension_score()` 计算伸展分数

`finger_extension_score()` 的计算逻辑：
- 计算指尖到掌心的距离与指关节到掌心的距离之差（`tip_to_center - pip_to_center`），差值越大说明手指越伸展
- 计算指尖沿手指骨骼方向的投影长度，作为辅助指标
- 两项加权求和得到伸展分数，超过阈值则判定为"伸展"

### 2.6 书写置信度计算

`compute_writing_confidence()` 综合多个因素输出 0-1 的连续置信度分数：

**正向因素**：
- 食指伸展分数（`index_score - 0.12`）乘以 1.15
- 食指指尖到掌心的相对距离（`tip_to_palm - 0.55`）乘以 0.55
- 食指锁定信任度（`index_trusted`）加 0.22
- 手部大小加成（最大 0.08）

**负向因素**：
- 中指、无名指、小指的伸展程度作为惩罚项（`curl_penalty`），乘以 0.30

计算公式：
```
confidence = max(0, index_score - 0.12) * 1.15
           + max(0, tip_to_palm - 0.55) * 0.55
           - curl_penalty * 0.30
           + (index_trusted ? 0.22 : 0)
           + min(0.08, hand_scale_px / 600.0)
```

最终值被裁剪到 [0, 1] 范围。

### 2.7 食指锁定与信任度

`compute_index_lock_trust()` 判断当前检测到的食指是否"可信"：

检查四个条件（全部满足才返回 true）：
1. 食指骨骼段 MCP→PIP→DIP→TIP 的方向一致性（`straight12 > 0.20`）
2. DIP→TIP 段的方向一致性（`straight23 > 0.10`）
3. 食指伸展程度（指尖到 MCP 距离 > PIP 到 MCP 距离 + 0.02）
4. 食指指尖与中指指尖的像素距离足够大（> `hand_scale * 0.10`）

这个判断的目的是在食指与其他手指靠得太近（如手指交叉）时标记为"不可信"，避免误检测。

### 2.8 TipLockState —— 食指位置锁定

`TipLockState` 是一个数据类，实现了食指位置的锁定和平滑机制。

**update() 方法**（正常帧处理）：

1. 如果尚未初始化，直接接受检测结果
2. 计算预测位置（上一帧位置 + 速度）
3. 计算检测结果与预测位置的偏差 `mismatch`
4. **可信且偏差在范围内**：直接采纳检测结果，重置不一致帧计数
5. **不可信或偏差过大**：
   - 不一致帧数 ≤ 4：沿历史轨迹预测（不采纳检测结果）
   - 不一致帧数 > 4：80% 预测 + 20% 检测结果的混合，缓慢重新接纳
6. 限幅：单帧移动距离不超过 `hand_scale * 0.50`
7. 速度更新：指数移动平均 `vx = vx * 0.45 + new_vx * 0.55`

**hold_predict() 方法**（手部丢失时）：

沿当前速度方向继续预测，速度衰减为 0.75 倍，移动距离上限为 `hand_scale * 0.30`。用于短时丢手保活。

### 2.9 TrackingState —— 整体追踪状态

`TrackingState` 管理所有追踪相关状态：

**书写状态机**（`update_drawing_state()`）：

```
IDLE → DRAW：confidence >= 0.38 且连续 2 帧确认
DRAW → IDLE：confidence < 0.25 且连续 6 帧确认
```

中间地带的处理：
- 从 DRAW 切回 IDLE 时，如果 confidence >= 0.22 且食指锁定仍然可信，不立即清零确认帧，而是缓慢递减，避免短暂的手指弯曲导致停笔

**光标更新**（`update_cursor()`）：

1. 先通过 `TipLockState` 处理食指位置锁定
2. 再通过 `SmoothedPoint` 做指数移动平均平滑
3. 平滑系数 alpha 根据移动距离动态调整：移动越快，alpha 越大（响应越快），范围 [0.24, 0.62]

**丢手处理**（`on_missing_hand()`）：

1. 递增丢手帧计数
2. 如果 TipLockState 仍然有效且丢手帧数 ≤ 2：沿历史轨迹预测，维持当前书写状态
3. 否则：重置所有状态，报告手部丢失

### 2.10 主循环

主循环的执行流程：

1. 从采集线程读取最新帧（带序列号去重）
2. 检查是否到达下一追踪时间点（`--tracking-fps` 控制，默认 45 FPS）
3. 如果帧宽度超过 `--detect-max-width`（默认 512），先缩小再送入检测线程
4. BGR 转 RGB，存入检测线程的共享变量
5. 从检测线程读取最新检测结果
6. 如果检测到手部：
   - 计算手部跨度（食指 MCP 到小指 MCP 的像素距离）
   - 检测手指状态和手势
   - 计算食指锁定信任度和书写置信度
   - 更新书写状态机和光标位置
7. 如果未检测到手部：调用 `on_missing_hand()` 尝试保活
8. 输出 JSON 到 stdout
9. 每 10 秒输出一次帧处理统计（已处理帧数、丢弃帧数、检测延迟）

### 2.11 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stdin-frames` | 关闭 | 从 stdin 读取 JPEG 帧（Qt 集成模式） |
| `--list-cameras` | 关闭 | 列出可用摄像头后退出 |
| `--max-cameras` | `6` | 探测摄像头的最大索引数 |
| `--camera-index` | `0` | 独立模式下的摄像头索引 |
| `--camera-name` | 空 | 独立模式下的 DirectShow 摄像头名称 |
| `--width` | `1280` | 独立模式下的采集宽度 |
| `--height` | `720` | 独立模式下的采集高度 |
| `--mirror` | 关闭 | 镜像翻转输入帧 |
| `--tracking-fps` | `45.0` | 追踪 JSON 输出帧率 |
| `--detect-max-width` | `512` | 送入检测的最大帧宽度（降采样） |

### 2.12 运行方式

**Qt 集成模式**（由 `AirWriteController` 自动启动）：

```powershell
python scripts/hand_tracker_service.py --stdin-frames --tracking-fps 45
```

**独立调试模式**：

```powershell
python scripts/hand_tracker_service.py --camera-index 0 --width 1280 --height 720
```

---

## 3. export_test_images.py —— 测试图片导出

**路径**：`scripts/project_operation/export_test_images.py`

**用途**：从 MNIST 测试集中导出若干张样本图片为 PNG 文件，用于 C++ 推理模块的离线验证。

### 3.1 工作原理

脚本使用 torchvision 的 `datasets.MNIST` 加载测试集，对每张图片执行 `ToTensor()` 转换后，再用 `ToPILImage()` 转回 PIL 格式并保存为 PNG。

导出的文件名格式为 `mnist_XX_label_Y.png`，其中 XX 是样本索引（两位数字），Y 是对应的数字标签（0-9）。这样在验证 C++ 推理时，可以直接通过文件名确认预期结果。

### 3.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-root` | `data` | MNIST 数据集存放目录 |
| `--output-dir` | `test_images` | 导出图片的输出目录 |
| `--count` | `10` | 导出图片数量 |
| `--mirror-url` | Google Cloud Storage | MNIST 下载镜像 |

### 3.3 使用示例

```powershell
py -3.12 scripts\project_operation\export_test_images.py --count 20 --output-dir test_images
```

执行后 `test_images/` 目录下会生成 20 张 28x28 的灰度 PNG 图片，每张图片的文件名包含其真实标签。这些图片可以直接被 C++ 端的 `cv::imread` 读取，送入 `DigitRecognizer::predict()` 验证推理结果是否正确。

---

## 依赖关系

三个脚本共同依赖的 Python 包：

| 包 | 用途 | 使用者 |
|----|------|--------|
| torch | PyTorch 深度学习框架 | train_mnist.py |
| torchvision | MNIST 数据集和图像变换 | train_mnist.py, export_test_images.py |
| opencv-python | 图像解码和摄像头采集 | hand_tracker_service.py |
| mediapipe | 手部关键点检测 | hand_tracker_service.py |
| numpy | 数组操作 | hand_tracker_service.py |

项目使用系统 Python 3.12，不创建新的虚拟环境。mediapipe 和 opencv-python 需要全局安装：

```powershell
pip install mediapipe opencv-python
```
