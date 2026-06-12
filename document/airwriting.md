# 隔空书写扩展开发说明

本文档记录当前项目的隔空书写实现。

---

## 目标

- 在 Qt 应用中实时显示清晰、低延迟的摄像头画面。
- 实时检测手部并稳定定位食指指尖。
- 将指尖轨迹同步映射到 Qt 画板，支持明确的起笔/停笔逻辑。
- 不影响原有数字识别流程（Canvas + LibTorch）。

---

## 技术栈

- UI 与画板：Qt Widgets + Canvas
- 摄像头采集与预览：Qt `QCamera` + `QVideoSink`
- 帧桥接：Qt 通过 `QProcess` stdin 将压缩后的 JPEG 帧送入 Python
- 手部检测：Python + OpenCV + MediaPipe Tasks `HandLandmarker`
- 识别层：C++ LibTorch `DigitRecognizer`

---

## 架构设计

### 画面归 Qt

Qt 负责唯一的摄像头实例与原生预览：

- `QCamera` 直接采集本地摄像头。
- `QVideoSink` 在 Qt 侧实时显示画面。
- 预览帧在 Qt 内部可选镜像，并可叠加手部指尖位置。

### 帧送 Python

Qt 将当前摄像头帧压缩成 JPEG，并通过 `QProcess` stdin 发送给 `scripts/hand_tracker_service.py`：

- 采用长度前缀协议，格式为 4 字节大端长度 + JPEG payload。
- Python 只负责消费这些帧并做手部推理，不再输出任何画面。
- 由于图像流不再从 Python 回传，旧方案中的卡死和轮询问题被移除。

### Python 返回数据

Python 进程通过 stdout 输出轻量 JSON：

```json
{
    "type": "frame",
    "has_hand": true,
    "drawing_active": true,
    "cursor": [320, 240],
    "frame_size": [640, 480],
    "confidence": 0.72,
    "gesture": 1,
    "index_trusted": true
}
```

Qt 根据 cursor 和 frame_size 把指尖轨迹映射到 Canvas，并在摄像头预览区域下方实时显示 `confidence`、`gesture`、`index_trusted` 三个追踪指标。

**gesture 取值含义：**

| 值 | 含义 | 说明 |
|----|------|------|
| 0 | IDLE | 非书写手势（握拳、张开手掌、其他手指组合） |
| 1 | DRAW | 仅食指抬起，其余手指弯曲 — 触发书写 |
| 2 | INDEX+MIDDLE | 食指和中指同时抬起 — 立即停止书写 |

**confidence（书写置信度）**：0-1 的连续分数，综合考虑食指伸展程度、其他手指弯曲程度和食指锁定信任度。值越高表示"正在写字"的判断越确定。状态机要求 confidence >= 0.38 连续 2 帧才从 IDLE 切到 DRAW。

**停止书写有两种路径：**
- 手势明确离开 DRAW（如伸出其他手指，gesture 变为 0 或 2）→ 立即停止，无需等待
- 手势仍为 DRAW 但检测信号不稳定（如手指抖动）→ confidence < 0.25 连续 6 帧后停止

**index_trusted（食指锁定信任度）**：布尔值，表示当前帧检测到的食指位置是否可靠。判断依据包括：
- 食指骨骼段（MCP→PIP→DIP→TIP）的方向一致性
- 食指伸展程度（指尖到 MCP 的距离 > PIP 到 MCP 的距离）
- 食指与中指指尖的间距（两根手指不应重叠）

当 index_trusted 为 false 时，追踪系统会沿历史轨迹预测食指位置，短时间不采纳新的检测结果，避免食指跳变到其他手指。

---

## 界面显示

摄像头预览区域下方有三个实时指标标签：

| 标签 | 数据来源 | 显示格式 | 颜色逻辑 |
|------|---------|---------|---------|
| 置信度 | `confidence` | `置信度: 72.0%` | 红(0%)→绿(100%) 渐变 |
| 手势 | `gesture` | `手势: DRAW` / `IDLE` / `INDEX+MID` | DRAW 绿色，其余灰色 |
| 锁定 | `index_trusted` | `锁定: 可信` / `不可信` | 可信绿色，不可信灰色 |

手部丢失时三个标签重置为 `--`。

---

## 稳定策略

Python 端实现了多层稳定机制：

- **限幅平滑**：对食指指尖位置做基于手型尺度的限幅平滑，每帧移动距离不超过动态上限。
- **手势状态机**：要求连续若干帧才能从 TRACK 切换到 DRAW 或从 DRAW 切回 TRACK，避免单帧误检。
- **食指锁定**：当检测到的食指位置与历史轨迹不一致时，短时间沿历史速度预测，不立即切换到异常点。
- **置信度滞回**：结合整手骨架、食指锁定和轨迹连续性判断，不再依赖单帧二值姿态。
- **短时丢手保活**：对 1-2 帧手部丢失做保活，降低突发中断概率。
- **停笔缓冲**：Qt 端在停笔时等待 120ms 确认，避免单帧误检把笔画切断。

---

## 起笔 / 停笔逻辑

手势判定逻辑：

- DRAW：仅食指抬起，其他手指弯曲。
- IDLE：其他手指组合。

Qt 侧行为：

- `drawing_active=true` 且当前无笔画时：`beginStroke()` 起笔。
- `drawing_active=true` 且已有笔画时：`appendStroke()` 连续绘制，超过 150 像素跳变时自动断笔。
- `drawing_active=false` 时先等待 120ms 确认，再 `endStroke()` 停笔。
- 手部丢失时立即结束当前笔画并重置稳定器。

坐标映射后还会经过一轮指数移动平均平滑（系数根据移动距离动态调整），进一步减少抖动。

---

## 关键文件

| 文件 | 职责 |
|------|------|
| `src/mainwindow.cpp` | Qt 摄像头预览、帧显示、坐标映射和画板绘制 |
| `src/airwritecontroller.cpp` | Python 追踪进程管理、stdin 帧发送和 stdout JSON 解析 |
| `scripts/hand_tracker_service.py` | 从 stdin 接收 JPEG 帧，执行 MediaPipe 追踪，输出指尖坐标和绘制状态 |

---

## 运行要求

- 系统 Python 3.12。
- 安装 `opencv-python`、`mediapipe`。
- 不创建新的虚拟环境。
- 首次启动隔空书写时，脚本会自动下载 MediaPipe hand landmarker 模型文件。

---

## 排查建议

- **摄像头无画面**：检查 Qt `QCamera` 是否成功启动，以及摄像头是否被其他程序占用。
- **追踪没有响应**：检查 Python 日志里是否出现 `Waiting for Qt frame stream` 或 MediaPipe 相关错误。
- **画面卡顿**：降低 Qt->Python 送帧频率或缩小送帧分辨率（`AirWriteController` 中的 `frameMaxWidth_` 和 `frameSubmitIntervalMs_`）。
- **轨迹抖动**：提高光照，或调整平滑系数与检测分辨率。
