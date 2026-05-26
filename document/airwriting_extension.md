# 隔空书写扩展开发说明

本文档记录当前项目的正式隔空书写实现。旧的临时文件轮询预览、Python 回传画面和 pinch 逻辑均已淘汰。

## 目标

- 在 Qt 应用中实时显示清晰、低延迟的摄像头画面。
- 实时检测手部并稳定定位食指指尖。
- 将指尖轨迹同步映射到 Qt 画板，支持明确的起笔/停笔逻辑。
- 不影响原有数字识别流程（Canvas + LibTorch）。

## 当前技术栈

- UI 与画板: Qt Widgets + Canvas
- 摄像头采集与预览: Qt `QCamera` + `QVideoSink`
- 帧桥接: Qt 通过 `QProcess` stdin 将压缩后的 JPEG 帧送入 Python
- 手部检测: Python + OpenCV + MediaPipe Tasks `HandLandmarker`
- 识别层: C++ LibTorch `DigitRecognizer`

## 新桥接架构

### 1. 画面归 Qt

Qt 负责唯一的摄像头实例与原生预览：

- `QCamera` 直接采集本地摄像头。
- `QVideoSink` 在 Qt 侧实时显示画面。
- 预览帧在 Qt 内部可选镜像，并可叠加手部指尖位置。

### 2. 帧送 Python

Qt 将当前摄像头帧压缩成 JPEG，并通过 `QProcess` stdin 发送给 `scripts/hand_tracker_service.py`：

- 采用长度前缀协议，格式为 `4 字节大端长度 + JPEG payload`。
- Python 只负责消费这些帧并做手部推理，不再输出任何画面。
- 由于图像流不再从 Python 回传，之前的卡死和轮询问题被移除。

### 3. Python 只回坐标

Python 进程只通过 stdout 输出轻量 JSON：

- `has_hand`
- `drawing_active`
- `cursor`
- `frame_size`

Qt 根据 cursor 和 frame_size 把指尖轨迹映射到 Canvas，并继续沿用平滑与断笔逻辑。

## 起笔 / 停笔逻辑

手势逻辑参考 `reference/aircatch`：

- DRAW: 仅食指抬起。
- IDLE: 其他组合。

Qt 侧行为：

- `drawing_active=true` 且当前无笔画时: `beginStroke()` 起笔。
- `drawing_active=true` 且已有笔画时: `appendStroke()` 连续绘制。
- `drawing_active=false` 或手部丢失时: `endStroke()` 停笔。

## 关键文件

- `src/mainwindow.cpp`
  - 管理 Qt 摄像头预览、帧显示、轨迹映射和画板绘制。
- `src/airwritecontroller.cpp`
  - 管理 Python 追踪进程、stdin 帧发送和 stdout JSON 解析。
- `scripts/hand_tracker_service.py`
  - 从 stdin 接收 JPEG 帧，执行 MediaPipe 追踪，仅输出食指指尖和绘制状态。

## 运行与构建

Release 构建命令：

```powershell
cmake --build build --config Release --target handwriting_recog
```

运行要求：

- 使用系统 Python 3.13。
- 安装 `opencv-python`、`mediapipe`。
- 不创建新的虚拟环境。

## 排查建议

- 摄像头无画面：检查 Qt `QCamera` 是否成功启动，以及摄像头是否被其他程序占用。
- 追踪没有响应：检查 Python 日志里是否出现 `Waiting for Qt frame stream` 或 MediaPipe 相关错误。
- 画面卡顿：降低 Qt→Python 送帧频率或缩小送帧分辨率。
- 轨迹抖动：提高光照，或调整平滑系数与检测分辨率。
