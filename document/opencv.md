# OpenCV 兼容层（已移除）

`src/opencv2/` 下的轻量级 API shim 已从项目中移除。

识别模块 `src/recognizer.cpp` 现在完全使用纯 Qt API（`QImage`）进行图像预处理，不再依赖任何 OpenCV 接口。Python 端的 `hand_tracker_service.py` 仍然使用 `opencv-python` 包（`cv2`），这与 C++ 端无关。
