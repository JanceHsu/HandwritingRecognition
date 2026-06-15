# 图像预处理流程详解

本文档详细描述 `src/recognizer.cpp` 中手写数字图像预处理的完整实现。预处理的目标是将用户在画板上书写的手写数字图像，转换为与 MNIST 训练数据相同格式的 28×28 归一化灰度图，以送入 TorchScript 模型进行推理。

---

## 总体流程概述

用户在 Qt 画板上书写数字后，程序获取画布的 `QImage`，经过以下五个步骤完成预处理：

1. **转为灰度图** — 将画布图像转为单通道 8 位灰度格式
2. **寻找笔迹边界** — 逐像素扫描，定位"墨迹"区域的最小外接矩形
3. **居中裁切为正方形** — 扩展边距、居中、裁切为正方形画布
4. **缩放到 28×28** — 双线性插值缩放至 MNIST 标准尺寸
5. **归一化** — 反转亮度（白底黑字→黑底白字）、减均值、除标准差

最终输出一个长度为 784（28×28）的 `std::vector<float>`，可直接转换为 PyTorch 张量。

---

## 调用入口

预处理由 `normalizeToMnist()` 函数完成，该函数定义在匿名命名空间中，被 `predict()` 和 `predictWithConfidence()` 两个公共方法调用。调用时首先将输入图像转为灰度格式，再传入预处理函数：

```cpp
// predict() 和 predictWithConfidence() 的开头部分
// 将用户输入的 QImage 转为 8 位灰度格式（单通道，每个像素 0~255）
const QImage gray = inputImage.convertToFormat(QImage::Format_Grayscale8);

// 调用预处理函数，返回长度为 784 的归一化浮点数组
std::vector<float> processed = normalizeToMnist(gray);

// 检查预处理结果的长度是否正确（28×28 = 784）
if (processed.size() != 28 * 28) {
    // 如果长度不正确，说明输入图像为空或处理失败，抛出异常
    throw std::runtime_error("Input image is empty");
}
```

`QImage::Format_Grayscale8` 格式每个像素用一个字节（`unsigned char`）表示亮度值，0 表示纯黑，255 表示纯白。转换过程中 Qt 会自动进行 RGB 到灰度的加权计算（标准公式：`0.299R + 0.587G + 0.114B`）。

---

## 常量定义

在预处理之前，文件顶部定义了 MNIST 数据集的统计常量，用于归一化步骤：

```cpp
// MNIST 训练集所有图片的像素均值（经过 0~1 归一化后的统计量）
// 这个值来源于对 MNIST 60000 张训练图片的全像素统计
constexpr float kMnistMean = 0.1307f;

// MNIST 训练集所有图片的像素标准差（经过 0~1 归一化后的统计量）
// 与均值配合使用，将像素分布标准化为近似标准正态分布
constexpr float kMnistStd = 0.3081f;
```

这两个值与 Python 端训练时使用的 `transforms.Normalize((0.1307,), (0.3081,))` 完全一致，确保推理时的输入分布与训练时相同。

---

## 完整预处理函数

以下是对 `recognizer.cpp` 中 `normalizeToMnist()` 函数的完整注释版本，逐行解释每个操作的目的和原理：

```cpp
// ============================================================
// normalizeToMnist — 将用户手写图像预处理为 MNIST 格式
//
// 输入：QImage 类型的灰度图像（Format_Grayscale8）
// 输出：长度为 784（28×28）的 float 向量，值已归一化
//
// 处理步骤：
//   1. 扫描图像找到笔迹的最小外接矩形（ink bounds）
//   2. 扩展边距并居中裁切为正方形
//   3. 缩放到 28×28 像素
//   4. 逐像素归一化（反转亮度 + 减均值除标准差）
// ============================================================
std::vector<float> normalizeToMnist(const QImage& grayImage)
{
    // ──────────────────────────────────────────────────────
    // 第二步：寻找笔迹边界
    //
    // 逐行逐列扫描整张灰度图，找出所有灰度值低于 245 的像素。
    // 245 是一个经验值：画板背景是纯白（255），而手写笔迹是黑色（0）。
    // 设定 245 作为阈值，可以有效区分"背景"和"墨迹"，同时容忍
    // 轻微的灰色笔迹（如抗锯齿产生的半透明边缘）。
    //
    // inkBounds 记录所有墨迹像素的最小外接矩形（左上角和右下角）。
    // ──────────────────────────────────────────────────────
    QRect inkBounds;  // 初始化为空矩形（isNull() == true）

    // 遍历图像的每一行（row 对应 Y 坐标）
    for (int row = 0; row < grayImage.height(); ++row) {
        // constScanLine(row) 返回第 row 行像素数据的只读指针
        // 返回类型为 const uchar*，每个字节代表一个像素的灰度值
        const auto* line = grayImage.constScanLine(row);

        // 遍历当前行的每一列（col 对应 X 坐标）
        for (int col = 0; col < grayImage.width(); ++col) {
            // 将像素值转为 unsigned char（0~255），与 245 比较
            // 小于 245 意味着该像素不是纯白/近白，属于墨迹区域
            if (static_cast<unsigned char>(line[col]) < 245) {
                // 如果 inkBounds 还是空矩形（这是找到的第一个墨迹像素），
                // 以当前像素位置初始化边界矩形（宽和高各为 1）
                if (inkBounds.isNull()) {
                    inkBounds = QRect(col, row, 1, 1);
                } else {
                    // 已有边界矩形，将其扩展以包含当前像素：
                    // 向左扩展（取当前左边界和当前列的较小值）
                    inkBounds.setLeft(std::min(inkBounds.left(), col));
                    // 向上扩展（取当前上边界和当前行的较小值）
                    inkBounds.setTop(std::min(inkBounds.top(), row));
                    // 向右扩展（取当前右边界和当前列的较大值）
                    inkBounds.setRight(std::max(inkBounds.right(), col));
                    // 向下扩展（取当前下边界和当前行的较大值）
                    inkBounds.setBottom(std::max(inkBounds.bottom(), row));
                }
            }
        }
    }
    // 扫描结束后，inkBounds 包含了所有墨迹像素的最小外接矩形

    // ──────────────────────────────────────────────────────
    // 第三步：扩展边距，居中裁切为正方形
    //
    // 为什么要扩展边距？直接紧贴墨迹裁切会导致数字"顶到边"，
    // 缩放到 28×28 后看起来比 MNIST 训练集中的数字更"满"，
    // 影响识别准确率。扩展 4 像素的边距模拟了 MNIST 数据集中
    // 数字周围的自然留白。
    //
    // 为什么要裁切为正方形？MNIST 图片是 28×28 的正方形。
    // 如果用户写的数字是扁长或瘦高的，不先转为正方形就直接缩放，
    // 会导致数字变形（横向压缩或纵向拉伸）。
    // ──────────────────────────────────────────────────────

    // adjusted(-4, -4, 4, 4)：在 inkBounds 的四条边各扩展 4 个像素
    // （left 减 4, top 减 4, right 加 4, bottom 加 4）
    // intersected(grayImage.rect())：与原图像矩形求交集，
    // 防止扩展后的边界超出图像范围（比如数字写在画布边缘时）
    inkBounds = inkBounds.adjusted(-4, -4, 4, 4).intersected(grayImage.rect());

    // 取边界矩形宽和高中较大的一边，作为正方形的边长
    // 这样做保证裁切时不会丢失任何墨迹像素
    const int side = std::max(inkBounds.width(), inkBounds.height());

    // 创建一个 side × side 的白色正方形画布
    // Format_Grayscale8：8 位灰度格式，与输入图像一致
    QImage padded(side, side, QImage::Format_Grayscale8);
    // 用白色（Qt::white 对应灰度值 255）填充整个画布
    padded.fill(Qt::white);

    // 创建 QPainter 用于在正方形画布上绘制裁切后的墨迹区域
    QPainter painter(&padded);

    // 计算墨迹在正方形画布中水平方向的居中偏移量：
    // (正方形边长 - 墨迹宽度) / 2 = 左侧需要留出的空白像素数
    const int offsetX = (side - inkBounds.width()) / 2;

    // 计算墨迹在正方形画布中垂直方向的居中偏移量：
    // (正方形边长 - 墨迹高度) / 2 = 上方需要留出的空白像素数
    const int offsetY = (side - inkBounds.height()) / 2;

    // grayImage.copy(inkBounds)：从原图中裁切出墨迹所在的矩形区域
    // drawImage(QPoint(offsetX, offsetY), ...)：将裁切区域绘制到
    // 正方形画布的居中位置。短边方向自然由白色填充补足。
    painter.drawImage(QPoint(offsetX, offsetY), grayImage.copy(inkBounds));

    // 结束绘制，释放 QPainter 持有的资源
    painter.end();

    // ──────────────────────────────────────────────────────
    // 第四步：缩放到 28×28
    //
    // MNIST 数据集中所有图片都是 28×28 像素。
    // 使用 Qt::SmoothTransformation（双线性/双三次插值）进行缩放，
    // 而非 Qt::FastTransformation（最近邻插值），
    // 因为平滑缩放产生的锯齿更少，图像质量更高。
    //
    // Qt::IgnoreAspectRatio：强制缩放到精确的 28×28，
    // 不保持宽高比（因为我们已经在上一步保证了输入是正方形，
    // 所以这里不会产生变形）。
    // ──────────────────────────────────────────────────────
    const QImage resized = padded.scaled(
        28,                           // 目标宽度：28 像素
        28,                           // 目标高度：28 像素
        Qt::IgnoreAspectRatio,        // 不保持宽高比（输入已是正方形）
        Qt::SmoothTransformation      // 使用平滑插值（双线性/双三次）
    );

    // ──────────────────────────────────────────────────────
    // 第五步：归一化
    //
    // 这一步完成三个转换：
    //
    // （a）亮度反转：画板是白底黑字（背景 255，笔迹接近 0），
    //      而 MNIST 是黑底白字（背景 0，笔迹接近 255）。
    //      公式 pixel = 1.0 - rawValue/255.0 将 255→0, 0→1，
    //      完成反转。
    //
    // （b）减去均值：减去 MNIST 训练集的均值 0.1307，
    //      使数据分布中心移到 0 附近。
    //
    // （c）除以标准差：除以 MNIST 训练集的标准差 0.3081，
    //      使数据分布的尺度与训练时一致。
    //
    // 综合公式：output = (pixel - 0.1307) / 0.3081
    // 其中 pixel = 1.0 - rawGrayValue / 255.0
    //
    // 这个归一化方式与 Python 训练代码中的
    // transforms.Normalize((0.1307,), (0.3081,)) 完全等价。
    // ──────────────────────────────────────────────────────

    // 创建长度为 784（28×28）的 float 向量，初始化为 0.0
    // 这是最终输出的归一化像素数组，按行优先顺序排列
    std::vector<float> normalized(28 * 28, 0.0f);

    // 遍历缩放后图像的每一行（0~27）
    for (int row = 0; row < resized.height(); ++row) {
        // 获取第 row 行像素数据的只读指针
        const auto* line = resized.constScanLine(row);

        // 遍历当前行的每一列（0~27）
        for (int col = 0; col < resized.width(); ++col) {
            // （a）亮度反转：将 0~255 的灰度值转为 0.0~1.0，
            //     然后用 1.0 减去它，完成白底黑字→黑底白字的反转。
            //     原始值 255（白色背景）→ 1.0 - 255/255 = 0.0（黑色背景）
            //     原始值 0（黑色笔迹）→ 1.0 - 0/255 = 1.0（白色笔迹）
            const float pixel = 1.0f - static_cast<float>(line[col]) / 255.0f;

            // （b）（c）减去 MNIST 均值并除以标准差，完成标准化
            //     确保输入数据的分布与模型训练时完全一致
            //     使用 static_cast<std::size_t> 将 int 索引转为无符号类型，
            //     避免编译器警告
            normalized[static_cast<std::size_t>(row) * 28 + static_cast<std::size_t>(col)] =
                (pixel - kMnistMean) / kMnistStd;
            // 存储位置：row * 28 + col，行优先排列
            // 第 0 行占索引 0~27，第 1 行占索引 28~55，以此类推
        }
    }

    // 返回归一化后的 784 维浮点向量
    // 调用方将使用 torch::from_blob 将其转换为形状 [1, 1, 28, 28] 的张量
    return normalized;
}
```

---

## 预处理结果到张量的转换

`normalizeToMnist()` 返回的 `std::vector<float>` 会立即被转换为 PyTorch 张量，送入模型推理。以下代码展示了从预处理结果到张量的完整过程：

```cpp
// 将预处理后的 float 数组转换为 PyTorch 张量
auto input = torch::from_blob(
    processed.data(),                // 指向 float 数组首地址的指针
    {1, 1, 28, 28},                  // 张量形状：
                                     //   第 0 维: batch_size = 1（一次识别一张图）
                                     //   第 1 维: channels = 1（灰度图，单通道）
                                     //   第 2 维: height = 28 像素
                                     //   第 3 维: width = 28 像素
    torch::TensorOptions().dtype(torch::kFloat32)  // 数据类型：32 位浮点数
).clone();
// .clone() 是必要的：from_blob 不拥有底层数据，它只是包装了一个指针。
// 如果不 clone，当 processed（std::vector<float>）离开作用域被销毁后，
// 张量会变成悬垂指针，读取到垃圾数据。clone 会复制一份数据到张量自己的内存中。

// 将张量传送到目标设备（CPU 或 CUDA GPU）
input = input.to(device);
```

---

## 预处理各步骤对识别效果的影响

每一步预处理都不是可有可无的，它们各自解决了特定的问题：

**灰度转换**：MNIST 是单通道灰度数据集。如果直接用 RGB 三通道图像输入，模型无法正确处理（输入维度不匹配）。即使人为复制为三通道，模型学到的单通道特征也无法适用。

**笔迹边界检测与居中裁切**：如果用户把数字写在画布的角落或边缘，不做裁切直接缩放到 28×28，数字会被压缩变形。以笔迹边界为基础居中裁切，保证数字在缩放后保持正确的宽高比和位置。报告第六节中提到，早期版本没有这一步，导致用户在画布边缘书写时识别错误。

**正方形化 + 4 像素边距**：MNIST 数据集中的数字都处于图片中央，周围有自然留白。4 像素的边距扩展模拟了这种留白，让输入图像的"密度"与训练数据一致。

**平滑缩放**：使用双线性/双三次插值而非最近邻插值，减少了缩放产生的锯齿和噪声，使笔迹边缘更平滑，更接近 MNIST 训练图片的风格。

**亮度反转**：画板是白底黑字，MNIST 是黑底白字。不反转的话，模型会把背景识别为笔迹、笔迹识别为背景，完全无法正确分类。

**减均值除标准差**：这是深度学习中标准的数据归一化手段。它将输入数据的分布标准化为均值接近 0、标准差接近 1，使得模型每一层的激活值都处于合理的数值范围内，加速收敛并提高数值稳定性。对于推理阶段，虽然没有收敛问题，但输入分布必须与训练时一致，否则模型内部的激活值会偏离训练时的分布，导致识别结果不准确。
