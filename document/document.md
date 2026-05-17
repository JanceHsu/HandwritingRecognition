# 手写数字识别系统（QT + PyTorch）单Agent 分阶段开发指导书

> 本指导书为 AI Agent 逐阶段执行而设计，需严格按照顺序完成，前一步的输出是后一步的依赖。
> 核心训练逻辑遵循 `mnist-handwriting-recognition` 技能，此处不再重复给出完整代码。
> 图形界面开发应当参考 `qt-cpp` 技能。
>
> 说明：当前仓库的正式构建与发布流程以 [document/build_workflow.md](build_workflow.md) 为准；本文档保留为历史分阶段设计记录，不再作为最新执行步骤。

------

## 项目目标

开发一个桌面应用程序：用户在画板上用鼠标书写数字，系统通过神经网络实时识别并显示结果。

- 训练 & 导出：Python + PyTorch（`mnist-handwriting-recognition` 技能）
- 推理：C++ LibTorch
- 界面：Qt6 Widgets (C++)
- 图像辅助：当前实现优先使用 Qt 图像处理，避免把识别点击路径绑定到本地 OpenCV 兼容层

------

## 阶段 0：环境准备与验证

**目标**：搭建可编译、可运行 Python 和 C++ 的开发环境。

1. **安装必需工具**
   - Python 3.8+、PyTorch（>=1.10）、torchvision
   - LibTorch（与 PyTorch **版本严格一致**，CPU 版即可）
   - Qt5（建议 5.15）、CMake（>=3.14）
   - OpenCV（供 C++ 使用）
2. **验证**
   - `python -c "import torch; print(torch.__version__)"` 正常输出
   - 编写并编译运行一个最小 Qt 窗口
   - 编写一个 C++ 程序，同时链接 LibTorch 和 OpenCV，成功执行并打印张量信息
3. **记录**
   - 所有组件的版本号，写入 `environment.txt` 备用

**产物**：`environment.txt`（版本清单）

------

## 阶段 1：模型训练与导出

**目标**：按照 `mnist-handwriting-recognition` 技能训练一个全连接网络，并将模型导出为 TorchScript 供 C++ 调用。

### 1.1 训练模型（执行技能流程）

- 严格遵循技能中的步骤：
  1. 使用 `torchvision.datasets.MNIST` 加载数据，`transforms.ToTensor()` 转为张量（像素 0‑1 范围）
  2. 构建包含 `Flatten` + 三层线性网络（含 ReLU）的 `NeuralNetwork`
  3. 使用 `CrossEntropyLoss` 和 `SGD(lr=1e-3)`，训练 50 个 epoch
- 训练过程中记录测试准确率，最终应达到约 92%（全连接网络基准）
- 保存模型状态字典（`model.pth`）作为中间产物，但最终交付物为 TorchScript 文件

### 1.2 导出 TorchScript

python

```
example = torch.rand(1, 1, 28, 28)   # 与 MNIST 尺寸一致
traced = torch.jit.trace(model.cpu(), example)
traced.save("mnist_model.pt")
```



**务必**在 CPU 上 trace，LibTorch 加载时设备保持一致。

### 1.3 明确预处理约定

- 训练时 `ToTensor()` 直接将 0‑255 整形像素转换为 0‑1 浮点，**未进行额外的反转或归一化**
- MNIST 图像为 **黑底白字**（背景 0，笔画接近 1）
- 用户画板为 **白底黑字**，因此推理时必须执行 **反转**（`1.0 - 归一化像素`）
- 这些约定将写入 `preprocess.json`，供 C++ 推理模块使用

**产物**：`mnist_model.pt`、`preprocess.json`（内容见下）

json

```
{
    "input_size": [1, 28, 28],
    "pixel_range": "0-1 float",
    "mnist_style": "black background, white digit",
    "invert_required": true
}
```



------

## 阶段 2：C++ 推理模块（LibTorch + OpenCV）

**目标**：封装模型加载、预处理和预测，提供独立可测的 `DigitRecognizer` 类。

### 2.1 类接口设计

文件：`src/recognizer.h`、`src/recognizer.cpp`

cpp

```
class DigitRecognizer {
public:
    explicit DigitRecognizer(const std::string& modelPath);
    int predict(const cv::Mat& inputImage);   // 输入任意尺寸灰度/彩色图
private:
    torch::jit::script::Module model;
    torch::Device device = torch::kCPU;
    cv::Mat preprocess(const cv::Mat& img);
};
```



### 2.2 预处理实现细节

严格按照 `preprocess.json` 约定：

1. 若输入为彩色，转换为灰度
2. 缩放至 **28×28**，使用 `cv::INTER_AREA` 插值
3. 转换为 `CV_32F`，除以 255.0 缩放到 [0,1]
4. **反转**：`cv::Mat` 上执行 `1.0 - img`
5. 确保数据在内存中连续（必要时 `img = img.clone()`），然后通过 `torch::from_blob` 创建形状为 `{1,1,28,28}` 的张量

### 2.3 推理函数

- 调用 `preprocess` 得到处理后的 `cv::Mat`
- 转换为 `torch::Tensor` 并送入模型
- 输出 `logits`，`argmax(1)` 获得预测标签
- 返回整数（0‑9）

### 2.4 单元测试（独立验证）

- 从 MNIST 测试集导出几张小图（黑底白字），用 `cv::imread` 读取
- 临时编写 `main` 函数加载 `mnist_model.pt`，调用 `predict`，确认结果与预期一致
- 测试完成后可删除独立 `main`，但保留推理类

**产物**：`recognizer.h/cpp`、测试截图或日志

------

## 阶段 3：Qt 图形界面（不接入识别）

**目标**：搭建美观、流畅的手写画板，为集成预留信号接口。

### 3.1 创建工程

- 使用 Qt Creator 新建 Qt Widgets Application，或手动编写文件：
  - `main.cpp`
  - `mainwindow.h` / `.cpp`
  - `canvas.h` / `.cpp`
- 可选用 `.ui` 文件设计界面，或纯代码布局

### 3.2 自定义绘图控件 `Canvas`

- 继承 `QWidget`，重写 `paintEvent` 和鼠标事件

- 内部维护一个 `QPixmap`（背景白色），用 `QPainter` 绘制黑色线条

- 画笔参数：粗细 18‑22px，圆头，抗锯齿

- 提供两个公共方法：

  cpp

  ```
  QPixmap getImage() const;   // 返回当前画布图像
  void clear();               // 清空为白底，刷新界面
  ```

  

- 鼠标移动时绘制线段并实时 `update()`

### 3.3 主窗口 `MainWindow` 布局

- 垂直/水平布局包含：
  - `Canvas` 控件（推荐 280×280 像素，方便缩放至 28×28）
  - 识别结果显示 `QLabel`（初始显示“识别结果：”）
  - `QPushButton`：“识别” 和 “清空”
- “清空”按钮连接 `Canvas::clear()`
- “识别”按钮的槽函数目前留空，或临时弹出提示“待集成”

### 3.4 图像转换辅助函数

在 `mainwindow.cpp` 中实现，为后续集成做准备：

cpp

```
cv::Mat MainWindow::pixmapToMat(const QPixmap& pix) {
    QImage img = pix.toImage().convertToFormat(QImage::Format_Grayscale8);
    return cv::Mat(img.height(), img.width(), CV_8UC1, const_cast<uchar*>(img.bits()), img.bytesPerLine()).clone();
}
```



**产物**：独立可运行的画板程序（不含识别），绘图流畅、界面清晰

------

## 阶段 4：系统集成与构建

**目标**：将推理模块嵌入界面，打通“书写 → 预处理 → 识别 → 显示”流程。

### 4.1 合并模块

1. 将 `recognizer.h/cpp` 加入 Qt 项目

2. 在 `MainWindow` 中增加成员 `DigitRecognizer* recognizer`

3. 在构造函数中初始化：

   cpp

   ```
   recognizer = new DigitRecognizer("path/to/mnist_model.pt");
   ```

   

   建议将模型文件放在可执行文件同目录，使用相对路径 `"./mnist_model.pt"`

4. 实现“识别”按钮的槽函数：

   cpp

   ```
   void MainWindow::onRecognize() {
       cv::Mat mat = pixmapToMat(canvas->getImage());
       if (mat.empty()) {
           // 提示“请先在画板上书写”
           return;
       }
       int digit = recognizer->predict(mat);
       resultLabel->setText(QString("识别结果: %1").arg(digit));
   }
   ```

   

### 4.2 构建配置 (CMakeLists.txt)

cmake

```
cmake_minimum_required(VERSION 3.14)
project(DigitRecognition)
set(CMAKE_CXX_STANDARD 14)

find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(digit_recog
    src/main.cpp
    src/mainwindow.cpp
    src/canvas.cpp
    src/recognizer.cpp
)

target_include_directories(digit_recog PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(digit_recog
    Qt5::Widgets
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
```



- 若使用 `.pro` 文件，需手动添加 LibTorch 的 `INCLUDEPATH` 和 `LIBS`，更推荐 CMake
- 编译前确保 `CMAKE_PREFIX_PATH` 包含 Qt5 和 LibTorch 的安装路径

### 4.3 运行与调试

- 将 `mnist_model.pt` 放在执行目录
- 书写数字 0‑9，检查识别结果
- 若识别错误，优先检查预处理（反转是否执行、缩放质量）

**产物**：完整可执行程序，核心功能可用

------

## 阶段 5：测试、优化与鲁棒性提升

**目标**：提升体验和泛化能力。

1. **功能测试**
   - 对每个数字（0‑9）书写 5‑10 种风格，记录识别准确率
   - 测试倾斜、大小变化、笔画断续等场景
2. **预处理优化（推荐）**
   - 在缩放前加入 **居中** 处理：找出笔画边界框，将数字居中后缩放，可大幅提高识别率
   - 代码示例思路：`cv::boundingRect` 获取非零区域，据此裁切并填充至正方形，再缩放
3. **性能优化**
   - 若识别卡顿，将推理放入独立线程（`QThread` 或 `QtConcurrent::run`），完成后通过信号更新 UI
   - 避免重复构造张量时的不必要拷贝
4. **异常保护**
   - 模型文件缺失时启动报错
   - 空画布识别时友好提示
   - 界面缩放时画板比例保持不变

**产物**：优化后的代码，测试记录（可记录于 `test_log.md`）

------

## 阶段 6：撰写实验报告

**目标**：生成结构完整的最终报告。

报告需包含以下部分（输出为 `experiment_report.md`）：

1. **实验背景与意义**（简述手写数字识别的应用）
2. **实验目标**
3. **方案设计**
   - 整体框架图（文字描述）
   - 模型结构：简单全连接网络（784→512→512→10）
   - 集成方案：TorchScript + LibTorch + Qt
4. **详细实现**
   - 训练过程（数据、超参、曲线）
   - C++ 推理模块核心预处理逻辑
   - Qt 界面设计与交互
5. **实验结果**
   - 界面截图（含识别实例）
   - 测试准确率（MNIST 测试集及自建手写样本）
   - 单次推理耗时
6. **问题与解决**（如预处理不当、性能问题等）
7. **总结与展望**

------

## 执行约定

- **严格顺序**：阶段 0 → 1 → 2 → 3 → 4 → 5 → 6
- **产出检查**：每阶段结束，确认产物完整后再继续
- **代码规范**：C++ 驼峰命名，Python 符合 PEP8，关键步骤加注释
- **路径一致性**：所有路径相对于工程根目录，便于移植
