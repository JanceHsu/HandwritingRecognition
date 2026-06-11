# 手写数字识别系统 — 分阶段开发指导书

> 本文档为项目的完整设计与开发指导书，按阶段记录从环境准备到最终交付的全流程。
> 各阶段的详细技术参数请参见对应的专题文档。

---

## 阶段 0：环境准备与验证

**目标**：搭建可编译、可运行 Python 和 C++ 的开发环境。

### 必要组件

- Python 3.12、PyTorch（>=2.0）、Torchvision
- LibTorch（与 PyTorch 版本严格匹配，CUDA 版）
- Qt 6.11.1 MSVC 2022 x64
- CMake 3.30+、CUDA Toolkit 12.8
- Visual Studio 2022 Build Tools

### 验证步骤

1. `py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` — 确认 PyTorch 和 CUDA 正常
2. 编写并编译运行一个最小 Qt 窗口 — 确认 Qt 工具链正常
3. 编写一个 C++ 程序，链接 LibTorch，执行 `torch::cuda::is_available()` — 确认 LibTorch CUDA 正常

详细的环境配置和构建流程参见 [build.md](build.md)。

---

## 阶段 1：模型训练与导出

**目标**：训练 MNIST 分类模型并导出为 TorchScript，供 C++ LibTorch 加载。

### 1.1 网络结构

项目使用小型卷积神经网络（Small CNN）：

```
特征提取：
  Conv2d(1,32,3x3) → BN → ReLU → Conv2d(32,32,3x3) → BN → ReLU → MaxPool(2) → Dropout2d(0.10)
  Conv2d(32,64,3x3) → BN → ReLU → Conv2d(64,64,3x3) → BN → ReLU → MaxPool(2) → Dropout2d(0.20)

分类：
  Flatten → Linear(3136,256) → ReLU → Dropout(0.3) → Linear(256,128) → ReLU → Dropout(0.2) → Linear(128,10)
```

输入为 1x28x28 灰度图像，输出为 10 个类别的 logits。

### 1.2 训练策略

- **数据增强**：随机旋转（+-15 度）、仿射变换（平移/缩放/错切）、透视变换、颜色抖动
- **优化器**：AdamW（lr=1e-3，weight_decay=5e-4）
- **学习率调度**：OneCycleLR（15% 预热，先升后降）
- **损失函数**：CrossEntropyLoss（label_smoothing=0.05）
- **GPU 优化**：混合精度训练（AMP）、TF32、cuDNN benchmark、channels_last 内存格式

### 1.3 导出

训练完成后保存两种格式：

- `model.pth`：PyTorch state_dict，供 Python 端使用
- `mnist_model.pt`：TorchScript 格式，供 C++ LibTorch 加载

导出时必须将模型移到 CPU，通过 `torch.jit.trace` 生成设备无关的计算图。

### 1.4 训练结果

在 RTX 5070 上训练 50 个 epoch，准确率约 99.7%。

详细的训练参数、代码和 CPU/GPU 对比参见 [model.md](model.md)。

---

## 阶段 2：C++ 推理模块

**目标**：封装模型加载、预处理和预测，提供独立可测的 `DigitRecognizer` 类。

### 2.1 类接口

```cpp
class DigitRecognizer {
public:
    explicit DigitRecognizer(const std::string& modelPath, bool useCuda);
    int predict(const QImage& inputImage);
    PredictResult predictWithConfidence(const QImage& inputImage);
    const std::string& deviceName() const;
    void warmUp();
private:
    torch::jit::script::Module model;
    torch::Device device;
    std::string deviceName_;
};
```

### 2.2 预处理流程

1. 转灰度图（3 通道/4 通道 → 单通道）
2. 找笔迹边界（灰度值 < 245 的区域）
3. 居中裁切为正方形（扩展 4 像素边距，白色填充）
4. 缩放到 28x28（cv::resize 或 QImage::scaled）
5. 归一化：`pixel = (1 - raw/255 - 0.1307) / 0.3081`

### 2.3 推理流程

1. 预处理得到 `std::vector<float>(784)`
2. `torch::from_blob` 创建 `[1,1,28,28]` 张量
3. `.to(device)` 迁移到 CPU 或 GPU
4. `model.forward()` 得到 logits
5. `argmax` 取预测标签，`softmax` 取置信度

### 2.4 设备管理

- 程序启动时检测 `torch::cuda::is_available()`，决定复选框是否可用
- 用户通过复选框切换 CPU/CUDA，切换时销毁当前识别器并重新加载模型
- CUDA 模式下加载后自动预热（一次空推理 + `torch::cuda::synchronize()`）

详细的推理技术细节参见 [model.md](model.md)。

---

## 阶段 3：Qt 图形界面

**目标**：搭建美观、流畅的手写画板和识别界面。

### 3.1 自定义绘图控件 Canvas

继承 `QWidget`，内部维护 `QPixmap` 作为底图：

- 鼠标按下：记录起始点，`beginStroke()`
- 鼠标移动：在上一个点和当前点之间画线段，`appendStroke()`
- 鼠标抬起：结束当前笔画，`endStroke()`
- 画笔：黑色、20px 粗、圆头、抗锯齿
- 支持程序化绘制（供隔空书写使用）

```cpp
void Canvas::drawLineTo(const QPoint& endPoint) {
    QPainter painter(&pixmap_);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(Qt::black, 20, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(lastPoint_, endPoint);
    update(QRect(lastPoint_, endPoint).normalized().adjusted(-24, -24, 24, 24));
    lastPoint_ = endPoint;
}
```

### 3.2 主窗口布局

左右分栏：

- **左侧**：上方隔空书写预览区（摄像头选择 + 预览画面 + 开关按钮），下方手写画布
- **右侧**：识别结果、可信度、隔空手势说明、摄像头镜像开关、CUDA 推理开关、操作提示、识别/清空按钮、操作日志

### 3.3 空画板检测

识别前检查画板是否有内容：扫描灰度值 < 245 的像素，少于 20 个时判定为空。

### 3.4 模型文件查找

搜索路径按优先级排列：`artifacts/models` → `dist/models` → exe 同级 `models/` → 上级 `models/`。确保在开发环境和发布环境中都能找到模型。

---

## 阶段 4：系统集成与构建

**目标**：将推理模块嵌入界面，打通完整流程，并配置好构建系统。

### 4.1 信号与槽连接

- "识别"按钮 → `onRecognize()`：获取画布图像 → `predictWithConfidence()` → 显示结果和可信度
- "清空"按钮 → `onClear()`：重置画布和结果标签
- CUDA 复选框 → `onCudaToggled()`：销毁识别器 → 以新设备重新加载
- 识别期间按钮锁定，防止重复点击重入

### 4.2 构建系统

项目同时维护两套构建入口：

- `CMakeLists.txt`：命令行 CMake 构建，用于发布
- `HandwritingRecognition.pro`：Qt Creator qmake 构建，用于调试

两者编译同一套源码，差异在 include/lib 路径配置和运行时部署。

### 4.3 LibTorch 与 Qt 宏冲突

LibTorch 的头文件和 Qt 的信号/槽机制都定义了 `slots` 和 `signals` 宏。`src/recognizer.h` 通过在包含 LibTorch 头文件前先 `#undef` 这两个宏、包含后再恢复的方式解决：

```cpp
#ifdef slots
#pragma push_macro("slots")
#undef slots
#define HANDWRITING_RECOGNITION_RESTORE_SLOTS_MACRO
#endif
// ... #include <torch/script.h> ...
#ifdef HANDWRITING_RECOGNITION_RESTORE_SLOTS_MACRO
#pragma pop_macro("slots")
#undef HANDWRITING_RECOGNITION_RESTORE_SLOTS_MACRO
#endif
```

### 4.4 CUDA 链接器标志

CMake 的 `find_package(Torch)` 自动注入 `-INCLUDE:?warp_size@cuda@at@@YAHXZ` 链接器选项，qmake 需要在 `.pro` 文件中手动添加。缺少该选项会导致 MSVC 链接器移除 CUDA 初始化入口，运行时无法检测到 CUDA。

详细的构建流程和差异参见 [build.md](build.md)。

---

## 阶段 5：隔空书写扩展

**目标**：通过摄像头实时捕捉手部运动，实现隔空书写数字。

### 5.1 架构设计

```
Qt 摄像头采集 → JPEG 压缩 → stdin → Python 进程（MediaPipe） → stdout JSON → Qt 映射到画布
```

关键决策：图像只从 Qt 流向 Python（单向），Python 只返回坐标 JSON，不传输图像。从根本上避免了旧方案中图像回传导致的管道阻塞。

### 5.2 Qt 端

- `QCamera` + `QVideoSink` 采集并显示摄像头画面
- `AirWriteController` 通过 `QProcess` 管理 Python 进程，发送 JPEG 帧（长度前缀协议）
- 坐标映射：线性缩放 + 指数移动平均平滑
- 停笔缓冲：120ms 确认，避免单帧误检切断笔画
- 跳变断笔：距离超过 150 像素时自动断笔

### 5.3 Python 端

- `hand_tracker_service.py`：MediaPipe HandLandmarker 检测 21 个手部关键点
- 三线程架构：采集线程、检测线程、主循环
- 手势判定：食指抬起 = DRAW，其他 = IDLE
- 书写置信度：综合食指伸展、其他手指弯曲、指尖锁定信任度，输出 0-1 连续分数
- 状态机：IDLE→DRAW 需 2 帧确认，DRAW→IDLE 需 6 帧确认
- 食指锁定：检测结果与历史轨迹不一致时，短时间沿历史速度预测
- 丢手保活：1-2 帧丢失时沿轨迹预测，不立即停笔

详细的隔空书写架构参见 [airwriting_extension.md](airwriting_extension.md)，Python 脚本参见 [python.md](python.md)。

---

## 阶段 6：测试与优化

### 6.1 功能测试

- 对每个数字（0-9）书写 5-10 种风格，记录识别准确率
- 测试倾斜、大小变化、笔画断续等场景
- 测试手部从画面消失再进入时笔画是否正常断开

### 6.2 隔空书写测试

- 均匀光照、干净背景下测试指尖追踪稳定性
- 书写数字 0-9 各 5 次，记录识别率
- 摄像头帧率 >= 15 fps

### 6.3 异常保护

- 模型文件缺失时启动报错
- 空画布识别时友好提示
- 摄像头未连接时禁用隔空模式
- 识别期间按钮锁定，防止重入

---

## 阶段 7：打包与发布

打包脚本 `scripts/package_release.ps1` 将构建产物整理为可分发的发布目录：

1. 复制 `handwriting_recog.exe` 到 `dist/`
2. 调用 `windeployqt` 部署 Qt 运行库
3. 复制 LibTorch DLL（CUDA 版含 `torch_cuda.dll` 等）
4. 复制模型文件到 `dist/models/{cpu,gpu}/`
5. 生成 `run_handwriting_recog.bat` 一键启动脚本

发布时应保留：`handwriting_recog.exe`、Qt 运行库、LibTorch 运行库、`models/`、`run_handwriting_recog.bat`。

详细的构建和发布流程参见 [build.md](build.md)。

---

## 源文件一览

| 文件 | 职责 |
|------|------|
| `src/main.cpp` | 程序入口，初始化 QApplication 和 MainWindow |
| `src/mainwindow.h/cpp` | 主窗口：UI 布局、摄像头预览、设备切换、识别调度、隔空书写 |
| `src/canvas.h/cpp` | 画布控件：鼠标绘制、程序化绘制（beginStroke/appendStroke/endStroke） |
| `src/recognizer.h/cpp` | 识别器：LibTorch 模型加载、QImage 预处理、推理、设备管理 |
| `src/airwritecontroller.h/cpp` | 隔空书写控制器：Python 进程管理、帧发送、JSON 解析 |
| `scripts/train_mnist.py` | MNIST 模型训练与 TorchScript 导出 |
| `scripts/hand_tracker_service.py` | 手部追踪服务（MediaPipe HandLandmarker） |
| `scripts/export_test_images.py` | 导出 MNIST 测试图片 |
| `CMakeLists.txt` | CMake 构建脚本 |
| `HandwritingRecognition.pro` | qmake 构建脚本 |
