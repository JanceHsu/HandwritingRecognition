# OpenCV 兼容层说明

`src/opencv2/` 目录下不是真正的 OpenCV 库，而是一套**轻量级的 API 兼容层**（shim）。它用纯 C++ 标准库实现了 `recognizer.cpp` 所需的 OpenCV 子集，使得项目不需要安装 OpenCV 即可编译。

两个头文件通过 `#include <opencv2/core.hpp>` 和 `#include <opencv2/imgproc.hpp>` 被引用——因为 `src/` 在编译器的 include 路径中，这些本地头文件会优先于系统安装的 OpenCV。

---

## core.hpp

提供基础数据类型和常量，对应 OpenCV 的 `opencv2/core.hpp`。

### 类型常量

| 常量 | 值 | 含义 |
|------|----|------|
| `CV_8UC1` | 0 | 单通道 8 位无符号整数（灰度图） |
| `CV_8UC3` | 1 | 三通道 8 位无符号整数（BGR 彩色图） |
| `CV_8UC4` | 2 | 四通道 8 位无符号整数（BGRA 带透明度） |
| `CV_32F` | 3 | 单通道 32 位浮点数 |
| `COLOR_BGR2GRAY` | 10 | BGR 转灰度的颜色空间转换代码 |
| `COLOR_BGRA2GRAY` | 11 | BGRA 转灰度的颜色空间转换代码 |
| `INTER_AREA` | 20 | 区域插值算法标识（缩小图像时使用） |

这些常量同时在 `cv` 命名空间和全局命名空间中定义，与 OpenCV 的使用习惯一致。

### cv::Size

简单的宽高结构体，包含 `width` 和 `height` 两个 `int` 成员，用于指定目标图像尺寸。

### cv::Scalar

四通道标量值，默认初始化为全零。构造时接受一个 `double` 值写入第一个通道。用于表示填充值、均值等标量数据。

### cv::Mat

核心矩阵类，是 OpenCV `cv::Mat` 的最小兼容实现。内部使用 `std::vector<uint8_t>` 存储像素数据，通过 `data` 公开指针供外部直接访问。

**构造方式**：

- `Mat()` — 空矩阵
- `Mat(rows, cols, type)` — 分配指定大小的内存并初始化为零
- `Mat(rows, cols, type, data, step)` — 从外部缓冲区拷贝数据，`step` 参数支持行对齐（stride），用于处理 QImage 等可能有行填充的图像数据

**支持的操作**：

| 方法 | 功能 |
|------|------|
| `empty()` | 是否为空（无数据或行列为零） |
| `rows()` / `cols()` | 行数和列数 |
| `channels()` | 通道数（由 type 推导） |
| `type()` | 类型标识（CV_8UC1 等） |
| `clone()` | 深拷贝，返回独立副本 |
| `convertTo(dst, rtype, alpha, beta)` | 类型转换，支持 `dst[i] = src[i] * alpha + beta` |
| `isContinuous()` | 始终返回 true（内部存储始终连续） |
| `elemSize()` / `step()` | 单个元素字节数和行步长 |

`convertTo()` 支持的转换路径：
- `CV_8UC1 -> CV_32F`：uint8 乘以 alpha 加 beta，输出 float
- `CV_32F -> CV_32F`：直接拷贝后乘以 alpha 加 beta
- `CV_8UC1 -> CV_8UC1`：直接拷贝
- `CV_32F -> CV_8UC1`：float 乘以 alpha 加 beta 后裁剪到 [0, 255] 并转为 uint8

**拷贝与移动语义**：支持拷贝构造、拷贝赋值、移动构造和移动赋值，移动后源对象的 `data` 指针被置空。

### operator-(float, Mat)

标量减矩阵操作。逐元素计算 `value - mat[i]`，支持 CV_32F 和 CV_8UC1 两种类型。用于实现预处理中的像素反转（`1.0 - img`）。

### cv::mean()

计算矩阵所有元素的均值，返回 `Scalar`。支持 CV_32F 和 CV_8UC1 类型。

---

## imgproc.hpp

提供图像处理函数，对应 OpenCV 的 `opencv2/imgproc.hpp`。依赖 `core.hpp`。

### cv::cvtColor(src, dst, code)

颜色空间转换。当前支持两种转换：

**BGR -> 灰度（`COLOR_BGR2GRAY`）**：输入必须是 3 通道。使用标准亮度公式：

```
gray = 0.114 * B + 0.587 * G + 0.299 * R
```

这是 ITU-R BT.601 标准的加权平均，绿色分量权重最大，因为人眼对绿色最敏感。

**BGRA -> 灰度（`COLOR_BGRA2GRAY`）**：输入必须是 4 通道。使用相同的亮度公式，忽略 Alpha 通道。

两种情况都是逐像素遍历，将三个（或四个）通道的值加权求和后写入单通道输出。结果裁剪到 [0, 255] 范围。

如果颜色代码不匹配或无法处理，直接克隆源矩阵作为输出。

### cv::resize(src, dst, size, fx, fy, interpolation)

图像缩放。将源图像缩放到 `size` 指定的目标尺寸。

当前实现使用**最近邻插值**（Nearest Neighbor）：对于目标图像中的每个像素 `(x, y)`，计算其在源图像中对应的位置 `(srcX, srcY)`，直接取最近的源像素值。计算公式：

```
srcX = floor((x + 0.5) * srcCols / dstCols)
srcY = floor((y + 0.5) * dstRows / srcRows)
```

坐标会被 `clamp` 到源图像的有效范围内，防止越界。

支持单通道（CV_8UC1）和多通道图像。单通道时直接拷贝单字节，多通道时逐通道拷贝。

`fx`、`fy`、`interpolation` 参数被声明但当前未使用——函数签名与 OpenCV 兼容，但内部始终使用最近邻插值。对于本项目的用途（将画布图像缩小到 28x28），最近邻插值已经足够。

---

## 在 recognizer.cpp 中的使用

`DigitRecognizer` 类通过 `#include <opencv2/imgproc.hpp>` 引入兼容层。`preprocess(const cv::Mat&)` 方法使用了以下 OpenCV API：

| 调用 | 所在步骤 | 用途 |
|------|---------|------|
| `cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY)` | 灰度转换 | 3 通道 BGR 转单通道灰度 |
| `cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY)` | 灰度转换 | 4 通道 BGRA 转单通道灰度 |
| `gray.rows()` / `gray.cols()` | 边界扫描 | 获取图像尺寸 |
| `reinterpret_cast<uint8_t*>(gray.data)` | 边界扫描 | 直接访问像素数据 |
| `cv::Mat padded(side, side, CV_8UC1)` | 居中裁切 | 创建白色填充的正方形画布 |
| `cv::resize(padded, resized, cv::Size(28, 28), ...)` | 缩放 | 缩放到 28x28 |
| `resized.empty()` / `resized.type()` | 校验 | 检查缩放结果有效性 |

`predict(const cv::Mat&)` 和 `predictWithConfidence(const cv::Mat&)` 两个公开接口接受 `cv::Mat` 输入。同时还有对应的 `predict(const QImage&)` 和 `predictWithConfidence(const QImage&)` 版本，使用纯 Qt API 实现相同的预处理逻辑，不依赖兼容层。

主窗口 `mainwindow.cpp` 中的实际识别调用走的是 `QImage` 路径，`cv::Mat` 路径保留作为兼容接口。
