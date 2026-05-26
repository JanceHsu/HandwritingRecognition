# 隔空手写数字识别扩展开发指导书（单Agent）

> 本指导书在原《手写数字识别系统（QT + PyTorch）单Agent分阶段开发指导书》基础上，追加**摄像头手部追踪隔空书写**功能。
> 实施前要求阶段0–4已完成，即拥有可运行的绘图识别程序。
> 本扩展遵循原有执行约定，各步骤严格按序执行。

> 当前仓库的默认实现已经迁移为 **Qt Widgets + Python 3.13 + OpenCV + MediaPipe Tasks**。Qt 侧使用原生 `QCamera`/`QVideoSink` 直接显示摄像头画面，并将压缩后的 JPEG 帧通过 `QProcess` stdin 送入 `scripts/hand_tracker_service.py`；Python 只输出整手 21 关键点、食指指尖游标和绘制状态，不再承担任何画面回传。本文件下方保留的 Windows Media Foundation / pinch 路线仅作为历史方案说明，不再是当前默认实现。

------

## 扩展目标

在现有桌面程序中增加**隔空书写模式**：

- 通过电脑摄像头实时捕捉手部运动
- 使用手部关键点检测算法追踪食指指尖
- 将指尖位置映射到画布，模拟鼠标绘制数字
- 用户可在空中写出数字并由原有模型进行识别

整体复用已有的 `Canvas` 绘图控件、`DigitRecognizer` 推理模块以及界面布局。

------

## 总体方案

1. **摄像头采集**：Qt 原生 `QCamera` 负责采集与显示，摄像头只由 Qt 占用一次，避免与 Python 抢占设备
2. **帧送入**：Qt 将摄像头帧压缩成 JPEG 后，通过 `QProcess` stdin 送给 Python，采用长度前缀协议避免粘包
3. **手部追踪**：Python 使用 MediaPipe Tasks `HandLandmarker` 获取整手 21 个关键点，直接读取食指指尖作为虚拟笔刷游标
4. **绘制状态**：以手部关键点关系判断是否允许落笔，取消“严格捏合才落笔 / 识别即落笔”双模式，统一为单一路径；Qt 用返回的 cursor 坐标映射到 `Canvas` 并叠加到预览上
5. **界面整合**：
   - 添加“隔空书写”开关按钮
    - 新增 `QLabel` 用于实时显示 Qt 原生摄像头画面，并在预览上叠加手部位置与轨迹
   - 绘制轨迹与识别按钮保持不变

------

## 阶段 E1：环境准备 – 集成 MediaPipe 与线程支持

**目标**：在现有C++项目中添加MediaPipe Hands C++库，并验证编译与运行。

### E1.1 安装 MediaPipe C++ SDK

- 参考 [MediaPipe官方C++安装指南](https://mediapipe.dev/)（推荐使用 v0.10.0 以上）

- 简要步骤（Linux/macOS示例，Windows可使用WSL或参照对应编译方式）：

  bash

  ```
  git clone https://github.com/google/mediapipe.git
  cd mediapipe
  # 安装bazel或使用预编译库方式（推荐使用预编译包）
  ```

  

- **推荐使用预编译的`mediapipe`C++库**，可从 [MediaPipe Releases](https://github.com/google/mediapipe/releases) 下载对应平台的`hand_tracking`图二进制文件及头文件。

- 将以下文件置于项目目录`third_party/mediapipe/`：

  - `mediapipe/framework` 头文件
  - `mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt` （计算图配置）
  - `mediapipe/modules/` 下相关模型文件（`hand_landmark.tflite`, `palm_detection.tflite` 等）

### E1.2 修改 CMakeLists.txt

在原有 `CMakeLists.txt` 中增加：

cmake

```
# MediaPipe 集成
set(MEDIAPIPE_DIR ${CMAKE_SOURCE_DIR}/third_party/mediapipe)
include_directories(${MEDIAPIPE_DIR})
# 链接 mediapipe 库（假设已编译为 libmediapipe.a 或 .so）
find_library(MEDIAPIPE_LIB mediapipe PATHS ${MEDIAPIPE_DIR}/lib)
target_link_libraries(handwriting_recog ${MEDIAPIPE_LIB} ...)  # 与其他库并列
```



确保 `mediapipe` 所需的第三方库（protobuf, abseil等）已正确配置。
建议使用 MediaPipe 提供的 `mediapipe_api` 封装或直接调用框架核心函数。

### E1.3 验证编译

- 新建一个简单的测试代码：初始化 MediaPipe 图，处理一张静态图片，输出手部关键点日志。
- 若成功编译并输出正常坐标，则环境搭建完成。

**产物**：可编译的工程，包含 MediaPipe 依赖，测试代码运行截图。

------

## 阶段 E2：摄像头采集与手部追踪模块

**目标**：实现 `HandTracker` 类，封装摄像头读取与 MediaPipe 推断，在独立线程中输出关键点。

### E2.1 摄像头捕获线程

新增文件 `src/cameraworker.h` / `.cpp`：

cpp

```
class CameraWorker : public QObject {
    Q_OBJECT
public:
    explicit CameraWorker(QObject *parent = nullptr);
    void start();   // 打开摄像头并启动定时器
    void stop();    // 停止采集

signals:
    void frameCaptured(const cv::Mat& frame);  // 原始帧
    void errorOccurred(const QString& msg);

private slots:
    void processFrame();

private:
    cv::VideoCapture cap;
    QTimer *timer;
};
```



- 在 `processFrame()` 中读取一帧，通过 `cv::cvtColor` 转为 RGB，然后发射信号 `frameCaptured`。

### E2.2 手部关键点推断

新增类 `HandLandmarkDetector`（文件 `src/handdetector.h/.cpp`）：

cpp

```
class HandLandmarkDetector {
public:
    HandLandmarkDetector(const std::string& graphConfigPath);
    // 输入 BGR 图像，返回 21 个关键点（归一化坐标 [0,1]），无手时返回空
    std::vector<cv::Point2f> detect(const cv::Mat& bgrFrame);

private:
    // 内部持有 MediaPipe 计算图实例
    // 略：具体使用 mediapipe::CalculatorGraph 等
};
```



- 实现中需加载 `hand_tracking_desktop_live.pbtxt` 配置。
- 若 MediaPipe C++ 接入复杂度太高，可使用 **OpenCV DNN 加载 MediaPipe 手部模型** 的备选方案（但仍建议使用原生MediaPipe以确保准确性）。
- 提取索引8（食指指尖）作为追踪点。

为降低难度，备用简单方案（鲁棒性差，仅作为逃生路径）：

- 肤色分割（HSV空间）+ 轮廓分析 + 凸包缺陷寻找指尖。
- 鉴于本项目为教学性质，推荐优先尝试 MediaPipe。

### E2.3 追踪器集成

在 `CameraWorker` 内部持有 `HandLandmarkDetector`，处理每帧后发射包含指尖像素坐标的信号：

cpp

```
signals:
    void fingertipPosition(const QPointF& pos); // 摄像头画面中的坐标
    void processedFrame(const QImage& frame);   // 画有标记点的帧，用于显示
```



**产物**：`cameraworker.h/.cpp`、`handdetector.h/.cpp` 及测试代码。

------

## 阶段 E3：坐标映射与画布绘制模拟

**目标**：将摄像头中的指尖坐标转换为画布坐标，在 `Canvas` 上“隔空”绘制。

### E3.1 坐标映射函数

在 `MainWindow` 中添加辅助方法：

cpp

```
QPointF MainWindow::mapCameraToCanvas(const QPointF& cameraPos,
                                      const QSize& cameraFrameSize,
                                      const QSize& canvasSize)
{
    // 简单线性映射，假设摄像头画面横向镜像，且映射区域为整个画面
    double xRatio = static_cast<double>(canvasSize.width()) / cameraFrameSize.width();
    double yRatio = static_cast<double>(canvasSize.height()) / cameraFrameSize.height();
    // 镜像翻转以符合自然书写方向（通常摄像头前置需要水平翻转）
    return QPointF((cameraFrameSize.width() - cameraPos.x()) * xRatio,
                   cameraPos.y() * yRatio);
}
```



- 可根据实际情况加入稳定区域（如只映射画面中央矩形区域）以减少误触。

### E3.2 隔空绘制状态管理

在 `MainWindow` 中添加以下成员：

cpp

```
bool m_airWritingActive = false;    // 是否启用隔空书写
QPointF m_lastFingerPos;           // 上一帧指尖位置
bool m_isDrawing = false;          // 是否正在绘制（指尖按下状态）
```



- 定义指尖“按下”判定：当指尖持续可见且某距离阈值内视为笔触接触。可简单认为一旦检测到手部关键点就绘制，丢失时提起。
- 为了避免重复起点，需要处理首帧初始化。

### E3.3 处理指尖信号

连接 `CameraWorker::fingertipPosition` 信号：

cpp

```
connect(worker, &CameraWorker::fingertipPosition, this, [this](const QPointF& camPos){
    if (!m_airWritingActive) return;
    QPointF canvasPos = mapCameraToCanvas(camPos, cameraFrameSize, canvas->size());
    if (m_isDrawing) {
        // 在 canvas 上绘制从 lastFingerPos 到 canvasPos 的线段
        canvas->drawLine(m_lastFingerPos, canvasPos);
    } else {
        // 新笔画起点
        canvas->startStroke(canvasPos);
        m_isDrawing = true;
    }
    m_lastFingerPos = canvasPos;
});
```



- 需要在 `Canvas` 类中添加 `drawLine(QPointF, QPointF)` 和 `startStroke(QPointF)` 公共接口，其内部实现与鼠标绘制类似（使用 `QPainter` 在 `QPixmap` 上绘制）。
- 当手部消失（未检测到关键点）时，应结束当前笔画，可设置 `m_isDrawing = false`。可通过定时器检测超时。

**产物**：修改后的 `Canvas` 类（支持程序化绘制）、`MainWindow` 坐标映射与信号处理逻辑。

------

## 阶段 E4：UI 集成与交互

**目标**：在主窗口添加摄像头预览和模式切换，实现完整的隔空书写体验。

### E4.1 界面调整

- 在原有布局中增加：
  - `QPushButton *btnAirMode`（“隔空书写”开关）
  - `QLabel *lblCameraPreview`（显示处理后的摄像头画面）
- 布局建议：左侧为摄像头画面（上方）和画板（下方），右侧为识别结果和按钮。根据窗口尺寸调整。

### E4.2 模式切换

按钮点击槽函数：

cpp

```
void MainWindow::toggleAirMode() {
    m_airWritingActive = !m_airWritingActive;
    if (m_airWritingActive) {
        cameraWorker->start();
        btnAirMode->setText("关闭隔空书写");
        // 可选：隐藏普通鼠标绘图提示
    } else {
        cameraWorker->stop();
        btnAirMode->setText("隔空书写");
    }
}
```



- 当关闭时，`Canvas` 恢复鼠标绘制行为；开启时，禁用鼠标绘制或同时保留（可叠加）。

### E4.3 摄像头画面显示

连接 `CameraWorker::processedFrame` 信号至 `lblCameraPreview` 的 `setPixmap`：

cpp

```
connect(worker, &CameraWorker::processedFrame, this, [this](const QImage& img){
    lblCameraPreview->setPixmap(QPixmap::fromImage(img).scaled(lblCameraPreview->size(),
                                                              Qt::KeepAspectRatio));
});
```



- `processedFrame` 信号由 `CameraWorker` 在得到指尖坐标后生成，用 `cv::circle` 在画面上绘制指尖标记，再转换为 `QImage` 发射。

### E4.4 识别按钮

- “识别”按钮逻辑不变，依然将当前 `Canvas` 内容转为 `QPixmap`，调用 `DigitRecognizer::predict()`。
- 可在隔空书写模式下自动识别（通过定时器），或保留手动点击。

**产物**：全功能界面，可通过摄像头隔空书写数字并识别。

------

## 阶段 E5：测试与优化

**目标**：确保追踪稳定，映射准确，体验流畅。

### E5.1 功能测试

- 在均匀光照、干净背景下测试指尖追踪是否稳定
- 书写数字 0‑9 各 5 次，记录识别准确率
- 测试手部从画面消失再进入时笔画是否正常断开和重新开始

### E5.2 性能与稳定性

- 保证摄像头帧率 ≥ 15 fps，若掉帧可降低捕获分辨率（如 640×480）
- 手部检测耗时较大时，使用独立推断线程，避免阻塞 UI
- 增加平滑处理（对指尖坐标做简单移动平均），减少抖动

### E5.3 异常处理

- 摄像头未连接时，弹出对话框并禁用隔空模式
- 当未检测到手部时，摄像头预览画面应显示“未检测到手部”提示
- 资源释放：程序退出时正确关闭摄像头和 MediaPipe 图

**产物**：经过优化的程序，测试记录（`airwriting_test_log.md`）。

------

## 执行约定

- 本扩展以原系统为基础，不可破坏现有鼠标书写功能
- 所有新增代码遵循原项目编码风格
- 涉及第三方库配置时，应在 `environment.txt` 中更新版本信息
- 阶段 E1–E5 必须顺序执行，每阶段完成后提交产物

------

> **附录：关键代码片段参考**
> 由于篇幅，此处仅给出关键接口，完整实现需Agent根据技能库自行填充。若MediaPipe集成困难，可暂时采用OpenCV肤色+轮廓指尖检测作为演示方案，并在报告中说明局限性。