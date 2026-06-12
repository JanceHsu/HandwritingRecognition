"""
Hand tracking service for Qt handwriting recognition.

Architecture:
  - Qt owns the webcam and displays the native preview.
  - Qt sends compressed frames to this process via stdin (length-prefixed JPEG).
  - MediaPipe HandLandmarker runs in a background thread.
  - This process sends lightweight JSON tracking data via stdout.

Stdout protocol (JSON lines):
  {"type":"frame","has_hand":true,"drawing_active":true,"frame_size":[1280,720],"cursor":[320,240]}
  {"type":"status","level":"info","message":"Hand tracker started"}
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
import mediapipe.tasks as mp_tasks


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent / ".cache" / "hand_landmarker.task"

WRIST = 0
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
LITTLE_MCP = 17
LITTLE_PIP = 18
LITTLE_TIP = 20


@dataclass
class SmoothedPoint:
    x: float = 0.0
    y: float = 0.0
    valid: bool = False

    def update(self, x: float, y: float, alpha: float = 0.40) -> tuple[float, float]:
        if not self.valid:
            self.x = x
            self.y = y
            self.valid = True
        else:
            self.x = self.x * (1.0 - alpha) + x * alpha
            self.y = self.y * (1.0 - alpha) + y * alpha
        return self.x, self.y


@dataclass
class TipLockState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    valid: bool = False
    inconsistent_frames: int = 0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.valid = False
        self.inconsistent_frames = 0

    def _predict(self) -> tuple[float, float]:
        return self.x + self.vx, self.y + self.vy

    def hold_predict(self, hand_scale: float) -> tuple[float, float]:
        if not self.valid:
            return 0.0, 0.0
        px, py = self._predict()
        max_step = max(10.0, hand_scale * 0.30)
        dx = px - self.x
        dy = py - self.y
        distance = math.hypot(dx, dy)
        if distance > max_step and distance > 0.0:
            scale = max_step / distance
            px = self.x + dx * scale
            py = self.y + dy * scale
        self.vx *= 0.75
        self.vy *= 0.75
        self.x = px
        self.y = py
        return self.x, self.y

    def update(self, raw_x: float, raw_y: float, hand_scale: float, trusted: bool) -> tuple[float, float]:
        if not self.valid:
            self.x = raw_x
            self.y = raw_y
            self.valid = True
            self.inconsistent_frames = 0
            return self.x, self.y

        predicted_x, predicted_y = self._predict()
        mismatch = math.hypot(raw_x - predicted_x, raw_y - predicted_y)
        mismatch_limit = max(20.0, hand_scale * 0.65)

        if trusted and mismatch <= mismatch_limit:
            target_x = raw_x
            target_y = raw_y
            self.inconsistent_frames = 0
        else:
            self.inconsistent_frames += 1
            if self.inconsistent_frames <= 4:
                target_x = predicted_x
                target_y = predicted_y
            else:
                blend = 0.20
                target_x = predicted_x * (1.0 - blend) + raw_x * blend
                target_y = predicted_y * (1.0 - blend) + raw_y * blend

        max_step = max(16.0, hand_scale * 0.50)
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        if distance > max_step and distance > 0.0:
            scale = max_step / distance
            target_x = self.x + dx * scale
            target_y = self.y + dy * scale

        new_vx = target_x - self.x
        new_vy = target_y - self.y
        self.vx = self.vx * 0.45 + new_vx * 0.55
        self.vy = self.vy * 0.45 + new_vy * 0.55
        self.x = target_x
        self.y = target_y
        return self.x, self.y


@dataclass
class TrackingState:
    cursor: SmoothedPoint = field(default_factory=SmoothedPoint)
    tip_lock: TipLockState = field(default_factory=TipLockState)
    drawing_active: bool = False
    draw_confirm_frames: int = 0
    idle_confirm_frames: int = 0
    missing_hand_frames: int = 0

    def reset(self) -> None:
        self.cursor.valid = False
        self.tip_lock.reset()
        self.drawing_active = False
        self.draw_confirm_frames = 0
        self.idle_confirm_frames = 0
        self.missing_hand_frames = 0

    def update_drawing_state(self, raw_active: bool, confidence: float) -> bool:
        if raw_active and confidence >= 0.38:
            self.draw_confirm_frames += 1
            self.idle_confirm_frames = 0
            if not self.drawing_active and self.draw_confirm_frames >= 2:
                self.drawing_active = True
        else:
            self.idle_confirm_frames += 1
            if confidence >= 0.22 and self.tip_lock.valid:
                self.draw_confirm_frames = max(0, self.draw_confirm_frames - 1)
            else:
                self.draw_confirm_frames = 0
            if self.drawing_active and self.idle_confirm_frames >= 6 and confidence < 0.25:
                self.drawing_active = False
        return self.drawing_active

    def update_cursor(self, raw_x: float, raw_y: float, hand_scale: float, trusted: bool) -> tuple[float, float]:
        lock_x, lock_y = self.tip_lock.update(raw_x, raw_y, hand_scale, trusted)
        if not self.cursor.valid:
            alpha = 0.36
        else:
            travel = math.hypot(lock_x - self.cursor.x, lock_y - self.cursor.y)
            alpha = min(0.62, max(0.24, 0.24 + travel / max(1.0, hand_scale) * 0.50))
        return self.cursor.update(lock_x, lock_y, alpha=alpha)

    def on_missing_hand(self, hand_scale: float) -> tuple[bool, float, float, bool]:
        self.missing_hand_frames += 1
        if self.tip_lock.valid and self.missing_hand_frames <= 2:
            hold_x, hold_y = self.tip_lock.hold_predict(hand_scale)
            cx, cy = self.cursor.update(hold_x, hold_y, alpha=0.42)
            return True, cx, cy, self.drawing_active
        self.reset()
        return False, 0.0, 0.0, False


def emit(payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def ensure_model_available() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    tmp_path = MODEL_PATH.with_suffix(".download")
    if tmp_path.exists():
        tmp_path.unlink()

    emit({"type": "status", "level": "info", "message": "正在下载 Hand Landmarker 模型。"})
    urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(MODEL_PATH)
    return MODEL_PATH


def open_camera(camera_name: str, camera_index: int) -> tuple[cv2.VideoCapture, str]:
    name = camera_name.strip()
    if name:
        capture = cv2.VideoCapture(f"video={name}", cv2.CAP_DSHOW)
        if capture is not None and capture.isOpened():
            return capture, f"name:{name}"
        if capture is not None:
            capture.release()

    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return capture, f"index:{camera_index}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qt hand tracking service")
    parser.add_argument("--stdin-frames", action="store_true", help="Read JPEG frames from stdin instead of opening camera")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices")
    parser.add_argument("--max-cameras", type=int, default=6, help="Maximum camera indices to probe")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to open in standalone mode")
    parser.add_argument("--camera-name", type=str, default="", help="DirectShow camera name to open in standalone mode")
    parser.add_argument("--width", type=int, default=1280, help="Capture width in standalone mode")
    parser.add_argument("--height", type=int, default=720, help="Capture height in standalone mode")
    parser.add_argument("--mirror", action="store_true", help="Mirror input frames before tracking")
    parser.add_argument("--tracking-fps", type=float, default=45.0, help="Tracking JSON output FPS")
    parser.add_argument("--detect-max-width", type=int, default=512, help="Downscale width for detection (0 disables)")
    return parser


def list_cameras(max_cameras: int) -> int:
    cameras: list[dict] = []
    for index in range(max(1, max_cameras)):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            continue
        ok, _ = capture.read()
        capture.release()
        if ok:
            cameras.append({"index": index, "name": f"Camera {index}"})
    emit({"type": "camera_list", "cameras": cameras})
    return 0


def vec_len(x: float, y: float) -> float:
    return math.hypot(x, y)


def normalize(x: float, y: float) -> tuple[float, float]:
    length = vec_len(x, y)
    if length <= 1e-6:
        return 0.0, 0.0
    return x / length, y / length


def estimate_hand_span(landmarks, width: int, height: int) -> float:
    if landmarks is None or len(landmarks) < 21:
        return 0.0

    index_mcp = landmarks[INDEX_MCP]
    little_mcp = landmarks[LITTLE_MCP]
    dx = (index_mcp.x - little_mcp.x) * width
    dy = (index_mcp.y - little_mcp.y) * height
    span = math.hypot(dx, dy)
    if span > 0.0:
        return span

    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    dx = (middle_mcp.x - wrist.x) * width
    dy = (middle_mcp.y - wrist.y) * height
    return math.hypot(dx, dy)


def finger_extension_score(landmarks, tip_idx: int, pip_idx: int, mcp_idx: int, palm_center: tuple[float, float], palm_span: float) -> float:
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]

    tip_to_center = math.hypot(tip.x - palm_center[0], tip.y - palm_center[1]) / palm_span
    pip_to_center = math.hypot(pip.x - palm_center[0], pip.y - palm_center[1]) / palm_span

    bone_x = pip.x - mcp.x
    bone_y = pip.y - mcp.y
    tip_x = tip.x - mcp.x
    tip_y = tip.y - mcp.y
    bone_len = vec_len(bone_x, bone_y)
    proj = 0.0
    if bone_len > 1e-6:
        proj = (tip_x * bone_x + tip_y * bone_y) / bone_len

    return (tip_to_center - pip_to_center) + (proj / max(1e-6, bone_len) - 1.0) * 0.35


def detect_finger_state(landmarks) -> tuple[bool, bool, bool, bool]:
    if landmarks is None or len(landmarks) < 21:
        return False, False, False, False

    palm_center = (
        (landmarks[WRIST].x + landmarks[INDEX_MCP].x + landmarks[MIDDLE_MCP].x + landmarks[RING_MCP].x + landmarks[LITTLE_MCP].x) / 5.0,
        (landmarks[WRIST].y + landmarks[INDEX_MCP].y + landmarks[MIDDLE_MCP].y + landmarks[RING_MCP].y + landmarks[LITTLE_MCP].y) / 5.0,
    )
    palm_span = max(
        math.hypot(landmarks[INDEX_MCP].x - landmarks[LITTLE_MCP].x, landmarks[INDEX_MCP].y - landmarks[LITTLE_MCP].y),
        math.hypot(landmarks[WRIST].x - landmarks[MIDDLE_MCP].x, landmarks[WRIST].y - landmarks[MIDDLE_MCP].y),
        0.05,
    )

    index_score = finger_extension_score(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, palm_center, palm_span)
    middle_score = finger_extension_score(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, palm_center, palm_span)
    ring_score = finger_extension_score(landmarks, RING_TIP, RING_PIP, RING_MCP, palm_center, palm_span)
    little_score = finger_extension_score(landmarks, LITTLE_TIP, LITTLE_PIP, LITTLE_MCP, palm_center, palm_span)

    return (
        index_score > 0.16,
        middle_score > 0.16,
        ring_score > 0.14,
        little_score > 0.12,
    )


def compute_drawing_state(index_up: bool, middle_up: bool, ring_up: bool, little_up: bool) -> int:
    if index_up and not middle_up and not ring_up and not little_up:
        return 1
    if index_up and middle_up and not ring_up and not little_up:
        return 2
    return 0


def compute_writing_confidence(landmarks, hand_scale_px: float, index_trusted: bool) -> float:
    if landmarks is None or len(landmarks) < 21:
        return 0.0

    palm_center = (
        (landmarks[WRIST].x + landmarks[INDEX_MCP].x + landmarks[MIDDLE_MCP].x + landmarks[RING_MCP].x + landmarks[LITTLE_MCP].x) / 5.0,
        (landmarks[WRIST].y + landmarks[INDEX_MCP].y + landmarks[MIDDLE_MCP].y + landmarks[RING_MCP].y + landmarks[LITTLE_MCP].y) / 5.0,
    )
    palm_span = max(
        math.hypot(landmarks[INDEX_MCP].x - landmarks[LITTLE_MCP].x, landmarks[INDEX_MCP].y - landmarks[LITTLE_MCP].y),
        math.hypot(landmarks[WRIST].x - landmarks[MIDDLE_MCP].x, landmarks[WRIST].y - landmarks[MIDDLE_MCP].y),
        0.05,
    )

    index_score = finger_extension_score(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, palm_center, palm_span)
    middle_score = finger_extension_score(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, palm_center, palm_span)
    ring_score = finger_extension_score(landmarks, RING_TIP, RING_PIP, RING_MCP, palm_center, palm_span)
    little_score = finger_extension_score(landmarks, LITTLE_TIP, LITTLE_PIP, LITTLE_MCP, palm_center, palm_span)

    curl_penalty = max(0.0, middle_score - 0.10) + max(0.0, ring_score - 0.08) + max(0.0, little_score - 0.06)
    tip_to_palm = math.hypot(landmarks[INDEX_TIP].x - palm_center[0], landmarks[INDEX_TIP].y - palm_center[1]) / palm_span
    confidence = (max(0.0, index_score - 0.12) * 1.15) + (max(0.0, tip_to_palm - 0.55) * 0.55) - (curl_penalty * 0.30)

    if index_trusted:
        confidence += 0.22
    confidence += min(0.08, max(0.0, hand_scale_px) / 600.0)

    return max(0.0, min(1.0, confidence))


def compute_index_lock_trust(landmarks, hand_scale_px: float) -> bool:
    if landmarks is None or len(landmarks) < 21:
        return False

    mcp = landmarks[INDEX_MCP]
    pip = landmarks[INDEX_PIP]
    dip = landmarks[INDEX_DIP]
    tip = landmarks[INDEX_TIP]
    middle_tip = landmarks[MIDDLE_TIP]

    v1x, v1y = normalize(pip.x - mcp.x, pip.y - mcp.y)
    v2x, v2y = normalize(dip.x - pip.x, dip.y - pip.y)
    v3x, v3y = normalize(tip.x - dip.x, tip.y - dip.y)

    straight12 = v1x * v2x + v1y * v2y
    straight23 = v2x * v3x + v2y * v3y

    tip_to_mcp = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
    pip_to_mcp = math.hypot(pip.x - mcp.x, pip.y - mcp.y)
    extension_gain = tip_to_mcp - pip_to_mcp

    index_middle_gap_px = math.hypot(tip.x - middle_tip.x, tip.y - middle_tip.y) * max(1.0, hand_scale_px)

    return (straight12 > 0.15 and straight23 > 0.08 and extension_gain > 0.01
            and index_middle_gap_px > max(6.0, hand_scale_px * 0.04))


def read_exact(stream, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        piece = stream.read(size - len(chunks))
        if not piece:
            return None
        chunks.extend(piece)
    return bytes(chunks)


def decode_jpeg(payload: bytes) -> np.ndarray | None:
    array = np.frombuffer(payload, dtype=np.uint8)
    if array.size == 0:
        return None
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame if frame is not None and frame.size > 0 else None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_cameras:
        return list_cameras(args.max_cameras)

    model_path = ensure_model_available()
    options = mp_tasks.vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.72,
        min_hand_presence_confidence=0.68,
        min_tracking_confidence=0.72,
    )
    landmarker = mp_tasks.vision.HandLandmarker.create_from_options(options)

    stop_event = threading.Event()
    capture_lock = threading.Lock()
    latest_capture_frame: dict = {
        "frame": None,
        "seq": 0,
        "timestamp_ms": 0,
    }

    detection_lock = threading.Lock()
    latest_detection_frame = {"frame": None, "timestamp_ms": 0}
    detection_results: dict = {
        "hand_landmarks": None,
        "last_error": None,
    }
    detection_stats = {"processed_frames": 0, "dropped_frames": 0, "last_latency_ms": 0.0}

    def capture_worker_camera() -> None:
        capture, opened_by = open_camera(args.camera_name, args.camera_index)
        if not capture.isOpened():
            hint = f" (name={args.camera_name})" if args.camera_name else ""
            emit({"type": "status", "level": "error", "message": f"打开摄像头 INDEX = {args.camera_index}{hint} 失败。"})
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emit({"type": "status", "level": "info", "message": f"帧源准备就绪 ({opened_by})"})

        try:
            while not stop_event.is_set():
                ok, raw_frame = capture.read()
                if not ok or raw_frame is None:
                    time.sleep(0.01)
                    continue

                if args.mirror:
                    raw_frame = cv2.flip(raw_frame, 1)

                with capture_lock:
                    latest_capture_frame["frame"] = np.ascontiguousarray(raw_frame)
                    latest_capture_frame["seq"] += 1
                    latest_capture_frame["timestamp_ms"] = int(time.perf_counter() * 1000)
        finally:
            capture.release()

    def capture_worker_stdin() -> None:
        stream = sys.stdin.buffer
        emit({"type": "status", "level": "info", "message": "正在等待 Qt 帧流……"})
        try:
            while not stop_event.is_set():
                header = read_exact(stream, 4)
                if header is None:
                    break
                payload_size = struct.unpack("!I", header)[0]
                if payload_size <= 0 or payload_size > 8 * 1024 * 1024:
                    continue
                payload = read_exact(stream, payload_size)
                if payload is None:
                    break
                frame = decode_jpeg(payload)
                if frame is None:
                    continue
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                with capture_lock:
                    latest_capture_frame["frame"] = frame
                    latest_capture_frame["seq"] += 1
                    latest_capture_frame["timestamp_ms"] = int(time.perf_counter() * 1000)
        except Exception as exc:
            emit({"type": "status", "level": "error", "message": f"标准输入帧流错误: {exc}"})

    def detection_worker() -> None:
        while not stop_event.is_set():
            frame_copy = None
            with detection_lock:
                if latest_detection_frame["frame"] is not None:
                    frame_copy = latest_detection_frame["frame"]
                    latest_detection_frame["frame"] = None

            if frame_copy is None:
                time.sleep(0.005)
                continue

            try:
                t0 = time.perf_counter()
                mp_img = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(frame_copy))
                results = landmarker.detect(mp_img)
                latency = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                with detection_lock:
                    detection_results["last_error"] = str(exc)
                time.sleep(0.01)
                continue

            with detection_lock:
                detection_results["hand_landmarks"] = results.hand_landmarks if results is not None and results.hand_landmarks else None
                detection_stats["processed_frames"] += 1
                detection_stats["last_latency_ms"] = latency

    producer_thread = threading.Thread(target=capture_worker_stdin if args.stdin_frames else capture_worker_camera, daemon=True)
    producer_thread.start()
    detector_thread = threading.Thread(target=detection_worker, daemon=True)
    detector_thread.start()

    tracking_state = TrackingState()
    tracking_interval = 1.0 / max(1.0, args.tracking_fps)
    next_tracking_at = 0.0
    last_timestamp_ms = 0
    last_detect_error_at = 0.0
    lost_frames = 0
    stats_last_emit = 0.0
    last_has_hand = False
    last_drawing_active = False
    last_cursor: list[int] | None = None
    last_confidence = 0.0
    last_gesture = 0
    last_index_trusted = False
    last_seen_capture_seq = 0

    emit({"type": "status", "level": "info", "message": "手部跟踪已启动。"})

    try:
        while True:
            with capture_lock:
                frame = latest_capture_frame["frame"]
                frame_seq = latest_capture_frame["seq"]
                frame_ts = latest_capture_frame["timestamp_ms"]

            if frame is None:
                lost_frames += 1
                if lost_frames == 1:
                    emit({"type": "status", "level": "warn", "message": "帧不可用。"})
                time.sleep(0.01)
                continue

            if frame_seq == last_seen_capture_seq:
                time.sleep(0.003)
                continue

            last_seen_capture_seq = frame_seq
            lost_frames = 0

            now = time.perf_counter()
            if now < next_tracking_at:
                time.sleep(0.002)
                continue

            detection_frame = frame
            if args.detect_max_width > 0 and frame.shape[1] > args.detect_max_width:
                scale = args.detect_max_width / frame.shape[1]
                new_h = max(1, int(round(frame.shape[0] * scale)))
                detection_frame = cv2.resize(frame, (args.detect_max_width, new_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            now_ms = int(now * 1000)
            ts = now_ms if now_ms > last_timestamp_ms else (last_timestamp_ms + 1)
            last_timestamp_ms = ts

            with detection_lock:
                if latest_detection_frame["frame"] is None:
                    latest_detection_frame["frame"] = np.copy(rgb)
                    latest_detection_frame["timestamp_ms"] = frame_ts if frame_ts > 0 else ts
                else:
                    detection_stats["dropped_frames"] += 1

                worker_landmarks = detection_results.get("hand_landmarks")
                worker_error = detection_results.get("last_error")
                if worker_error is not None:
                    if now - last_detect_error_at >= 1.0:
                        emit({"type": "status", "level": "warn", "message": f"MediaPipe 错误: {worker_error}"})
                        last_detect_error_at = now
                    detection_results["last_error"] = None

            h, w = frame.shape[:2]
            hand_scale = max(24.0, min(float(max(h, w)), float(min(h, w)) * 0.8))

            if worker_landmarks is not None and len(worker_landmarks) > 0:
                hand_landmarks = worker_landmarks[0]
                hand_scale = max(24.0, estimate_hand_span(hand_landmarks, w, h))
                tracking_state.missing_hand_frames = 0

                index_up, middle_up, ring_up, little_up = detect_finger_state(hand_landmarks)
                gesture = compute_drawing_state(index_up, middle_up, ring_up, little_up)

                raw_index_tip = hand_landmarks[INDEX_TIP]
                raw_x = float(raw_index_tip.x * w)
                raw_y = float(raw_index_tip.y * h)
                index_trusted = compute_index_lock_trust(hand_landmarks, hand_scale)

                writing_confidence = compute_writing_confidence(hand_landmarks, hand_scale, index_trusted)

                if gesture == 1:
                    writing_confidence = min(1.0, writing_confidence + 0.18)
                elif index_up and index_trusted:
                    writing_confidence = min(1.0, writing_confidence + 0.10)

                drawing_active = tracking_state.update_drawing_state(
                    gesture == 1 or (index_up and index_trusted),
                    writing_confidence,
                )

                cx, cy = tracking_state.update_cursor(raw_x, raw_y, hand_scale, index_trusted)

                last_has_hand = True
                last_drawing_active = drawing_active
                last_cursor = [int(round(cx)), int(round(cy))]
                last_confidence = round(writing_confidence, 3)
                last_gesture = gesture
                last_index_trusted = index_trusted
            else:
                has_hand, hold_x, hold_y, draw_state = tracking_state.on_missing_hand(hand_scale)
                last_has_hand = has_hand
                last_drawing_active = draw_state
                last_cursor = [int(round(hold_x)), int(round(hold_y))] if has_hand else None
                last_confidence = 0.0
                last_gesture = 0
                last_index_trusted = False

            next_tracking_at = now + tracking_interval

            payload = {
                "type": "frame",
                "has_hand": last_has_hand,
                "drawing_active": last_drawing_active,
                "frame_size": [frame.shape[1], frame.shape[0]],
                "cursor": last_cursor,
                "confidence": last_confidence,
                "gesture": last_gesture,
                "index_trusted": last_index_trusted,
            }
            emit(payload)

            if now - stats_last_emit >= 10.0:
                stats_last_emit = now
                with detection_lock:
                    stats = detection_stats.copy()
                emit({
                    "type": "status",
                    "level": "info",
                    "message": f"帧流处理监测: 已处理 {stats['processed_frames']} 帧，丢弃 {stats['dropped_frames']} 帧，延迟 {stats['last_latency_ms']:.1f} ms。",
                })

    except KeyboardInterrupt:
        emit({"type": "status", "level": "info", "message": "手部跟踪已停止。"})
    finally:
        stop_event.set()
        if not args.stdin_frames:
            try:
                producer_thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            detector_thread.join(timeout=1.0)
        except Exception:
            pass
        landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
