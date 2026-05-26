"""
Hand tracking service for Qt handwriting recognition.

Architecture:
  - Qt owns the webcam and displays the native preview
  - Qt sends compressed frames to this process via stdin (length-prefixed JPEG)
  - MediaPipe HandLandmarker runs in a background thread
  - This process sends only lightweight JSON tracking data via stdout
  - No image frames are sent back to Qt

Stdout protocol (JSON lines):
  {"type":"frame","has_hand":true,"drawing_active":true,"frame_size":[1280,720],"cursor":[320,240]}
  {"type":"status","level":"info","message":"Hand tracker started"}
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import threading
from mediapipe import Image, ImageFormat
import mediapipe.tasks as mp_tasks


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent / ".cache" / "hand_landmarker.task"

FINGER_INDEX_TIP = 8
FINGER_INDEX_PIP = 6
FINGER_MIDDLE_TIP = 12
FINGER_MIDDLE_PIP = 10
FINGER_RING_TIP = 16
FINGER_RING_PIP = 14
FINGER_LITTLE_TIP = 20
FINGER_LITTLE_PIP = 18


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

    emit({"type": "status", "level": "info", "message": "Downloading hand landmarker model…"})
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
    parser.add_argument("--tracking-fps", type=float, default=30.0, help="Tracking JSON output FPS")
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


def detect_finger_state(landmarks) -> tuple[bool, bool, bool, bool]:
    if landmarks is None or len(landmarks) < 21:
        return False, False, False, False

    def is_up(tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    index_up = is_up(FINGER_INDEX_TIP, FINGER_INDEX_PIP)
    middle_up = is_up(FINGER_MIDDLE_TIP, FINGER_MIDDLE_PIP)
    ring_up = is_up(FINGER_RING_TIP, FINGER_RING_PIP)
    little_up = is_up(FINGER_LITTLE_TIP, FINGER_LITTLE_PIP)
    return index_up, middle_up, ring_up, little_up


def compute_drawing_state(index_up: bool, middle_up: bool, ring_up: bool, little_up: bool) -> int:
    if index_up and not middle_up and not ring_up and not little_up:
        return 1
    if index_up and middle_up and not ring_up and not little_up:
        return 2
    return 0


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
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
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
            emit({"type": "status", "level": "error", "message": f"Failed to open camera index {args.camera_index}{hint}"})
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emit({"type": "status", "level": "info", "message": f"Frame source ready ({opened_by})"})

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
        emit({"type": "status", "level": "info", "message": "Waiting for Qt frame stream"})
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
            emit({"type": "status", "level": "error", "message": f"stdin frame stream error: {exc}"})

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

    smoothed_cursor = SmoothedPoint()
    tracking_interval = 1.0 / max(1.0, args.tracking_fps)
    next_tracking_at = 0.0
    last_timestamp_ms = 0
    last_detect_error_at = 0.0
    lost_frames = 0
    stats_last_emit = 0.0
    last_has_hand = False
    last_drawing_active = False
    last_cursor: list[int] | None = None
    last_seen_capture_seq = 0

    emit({"type": "status", "level": "info", "message": "Hand tracker started"})

    try:
        while True:
            with capture_lock:
                frame = latest_capture_frame["frame"]
                frame_seq = latest_capture_frame["seq"]
                frame_ts = latest_capture_frame["timestamp_ms"]

            if frame is None:
                lost_frames += 1
                if lost_frames == 1:
                    emit({"type": "status", "level": "warn", "message": "Frame unavailable"})
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
                        emit({"type": "status", "level": "warn", "message": f"MediaPipe error: {worker_error}"})
                        last_detect_error_at = now
                    detection_results["last_error"] = None

            if worker_landmarks is not None and len(worker_landmarks) > 0:
                hand_landmarks = worker_landmarks[0]
                h, w = frame.shape[:2]

                landmark_points = [[int(landmark.x * w), int(landmark.y * h)] for landmark in hand_landmarks]
                index_up, middle_up, ring_up, little_up = detect_finger_state(hand_landmarks)
                gesture = compute_drawing_state(index_up, middle_up, ring_up, little_up)
                drawing_active = gesture == 1
                index_tip = landmark_points[FINGER_INDEX_TIP]
                cx, cy = smoothed_cursor.update(float(index_tip[0]), float(index_tip[1]))

                last_has_hand = True
                last_drawing_active = drawing_active
                last_cursor = [int(round(cx)), int(round(cy))]
            else:
                last_has_hand = False
                last_drawing_active = False
                last_cursor = None
                smoothed_cursor.valid = False

            next_tracking_at = now + tracking_interval

            payload = {
                "type": "frame",
                "has_hand": last_has_hand,
                "drawing_active": last_drawing_active,
                "frame_size": [frame.shape[1], frame.shape[0]],
                "cursor": last_cursor,
            }
            emit(payload)

            if now - stats_last_emit >= 10.0:
                stats_last_emit = now
                with detection_lock:
                    stats = detection_stats.copy()
                emit({"type": "status", "level": "info", "message": f"detector: processed={stats['processed_frames']} dropped={stats['dropped_frames']} latency={stats['last_latency_ms']:.1f}ms"})

    except KeyboardInterrupt:
        emit({"type": "status", "level": "info", "message": "Hand tracker stopped"})
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
"""
Hand tracking service for Qt handwriting recognition.

Architecture:
  - Qt owns the camera and sends JPEG frames to this process over stdin.
  - Python runs MediaPipe HandLandmarker and emits only lightweight JSON via stdout.
  - No preview images are sent back to Qt.

Stdout protocol (JSON lines):
  {"type":"frame","has_hand":true,"drawing_active":true,"frame_size":[1280,720],"cursor":[320,240]}
  {"type":"status","level":"info","message":"Hand tracker started"}
"""

import argparse
import json
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import threading
from mediapipe import Image, ImageFormat
import mediapipe.tasks as mp_tasks


HAND_CONNECTIONS = mp_tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent / ".cache" / "hand_landmarker.task"

FINGER_INDEX_TIP = 8
FINGER_INDEX_PIP = 6
FINGER_MIDDLE_TIP = 12
FINGER_MIDDLE_PIP = 10
FINGER_RING_TIP = 16
FINGER_RING_PIP = 14
FINGER_LITTLE_TIP = 20
FINGER_LITTLE_PIP = 18


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


def emit(payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()


def read_exact(stream, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_jpeg_frame(stream) -> np.ndarray | None:
    header = read_exact(stream, 4)
    if header is None:
        return None

    payload_size = struct.unpack("!I", header)[0]
    if payload_size <= 0 or payload_size > 8 * 1024 * 1024:
        return None

    payload = read_exact(stream, payload_size)
    if payload is None:
        return None

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def ensure_model_available() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    tmp_path = MODEL_PATH.with_suffix(".download")
    if tmp_path.exists():
        tmp_path.unlink()

    emit({"type": "status", "level": "info", "message": "Downloading hand landmarker model…"})
    urlretrieve(MODEL_URL, tmp_path)
    tmp_path.replace(MODEL_PATH)
    return MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qt hand tracking service")
    parser.add_argument("--stdin-frames", action="store_true", help="Read JPEG frames from stdin")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to open in standalone mode")
    parser.add_argument("--camera-name", type=str, default="", help="DirectShow camera name to open in standalone mode")
    parser.add_argument("--width", type=int, default=960, help="Capture width in standalone mode")
    parser.add_argument("--height", type=int, default=540, help="Capture height in standalone mode")
    parser.add_argument("--mirror", action="store_true", help="Mirror incoming frames")
    parser.add_argument("--tracking-fps", type=float, default=30.0, help="Tracking JSON output FPS")
    parser.add_argument("--detect-max-width", type=int, default=512, help="Downscale width for detection (0 disables)")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices")
    parser.add_argument("--max-cameras", type=int, default=8, help="Maximum camera indices to probe")
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


def detect_finger_state(landmarks) -> tuple[bool, bool, bool, bool]:
    if landmarks is None or len(landmarks) < 21:
        return False, False, False, False

    def is_up(tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    return (
        is_up(FINGER_INDEX_TIP, FINGER_INDEX_PIP),
        is_up(FINGER_MIDDLE_TIP, FINGER_MIDDLE_PIP),
        is_up(FINGER_RING_TIP, FINGER_RING_PIP),
        is_up(FINGER_LITTLE_TIP, FINGER_LITTLE_PIP),
    )


def compute_drawing_state(index_up: bool, middle_up: bool, ring_up: bool, little_up: bool) -> int:
    if index_up and not middle_up and not ring_up and not little_up:
        return 1
    if index_up and middle_up and not ring_up and not little_up:
        return 2
    return 0


def iter_frames_from_stdin() -> tuple[np.ndarray | None, str]:
    frame = recv_jpeg_frame(sys.stdin.buffer)
    return frame, "stdin"


def run_frame_source(args, stop_event: threading.Event, latest_frame: dict, capture_lock: threading.Lock) -> None:
    if args.stdin_frames:
        while not stop_event.is_set():
            frame = recv_jpeg_frame(sys.stdin.buffer)
            if frame is None:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            with capture_lock:
                latest_frame["frame"] = np.ascontiguousarray(frame)
                latest_frame["seq"] += 1
                latest_frame["timestamp_ms"] = int(time.perf_counter() * 1000)
        return

    capture, _ = open_camera(args.camera_name, args.camera_index)
    if not capture.isOpened():
        emit({"type": "status", "level": "error", "message": f"Failed to open camera index {args.camera_index}"})
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            with capture_lock:
                latest_frame["frame"] = np.ascontiguousarray(frame)
                latest_frame["seq"] += 1
                latest_frame["timestamp_ms"] = int(time.perf_counter() * 1000)
    finally:
        capture.release()


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
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    landmarker = mp_tasks.vision.HandLandmarker.create_from_options(options)

    stop_event = threading.Event()
    capture_lock = threading.Lock()
    latest_capture_frame: dict = {"frame": None, "seq": 0, "timestamp_ms": 0}

    producer_thread = threading.Thread(
        target=run_frame_source,
        args=(args, stop_event, latest_capture_frame, capture_lock),
        daemon=True,
    )
    producer_thread.start()

    detection_lock = threading.Lock()
    latest_detection_frame = {"frame": None, "timestamp_ms": 0}
    detection_results: dict = {"hand_landmarks": None, "last_error": None}
    detection_stats = {"processed_frames": 0, "dropped_frames": 0, "last_latency_ms": 0.0}

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
                detection_results["hand_landmarks"] = results.hand_landmarks if results and results.hand_landmarks else None
                detection_stats["processed_frames"] += 1
                detection_stats["last_latency_ms"] = latency

    detector_thread = threading.Thread(target=detection_worker, daemon=True)
    detector_thread.start()

    smoothed_cursor = SmoothedPoint()
    tracking_interval = 1.0 / max(1.0, args.tracking_fps)
    next_tracking_at = 0.0
    last_timestamp_ms = 0
    last_detect_error_at = 0.0
    last_seen_capture_seq = 0
    stats_last_emit = 0.0

    last_has_hand = False
    last_drawing_active = False
    last_cursor: list[int] | None = None

    mode_desc = "stdin" if args.stdin_frames else f"index={args.camera_index}"
    emit({"type": "status", "level": "info", "message": f"Hand tracker started ({mode_desc})"})

    try:
        while True:
            with capture_lock:
                frame = latest_capture_frame["frame"]
                frame_seq = latest_capture_frame["seq"]
                frame_ts = latest_capture_frame["timestamp_ms"]

            if frame is None:
                time.sleep(0.01)
                continue
            if frame_seq == last_seen_capture_seq:
                time.sleep(0.003)
                continue
            last_seen_capture_seq = frame_seq

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
                        emit({"type": "status", "level": "warn", "message": f"MediaPipe error: {worker_error}"})
                        last_detect_error_at = now
                    detection_results["last_error"] = None

            if worker_landmarks is not None and len(worker_landmarks) > 0:
                hand_landmarks = worker_landmarks[0]
                h, w = frame.shape[:2]
                landmark_points = [[int(landmark.x * w), int(landmark.y * h)] for landmark in hand_landmarks]

                index_up, middle_up, ring_up, little_up = detect_finger_state(hand_landmarks)
                drawing_active = compute_drawing_state(index_up, middle_up, ring_up, little_up) == 1

                index_tip = landmark_points[FINGER_INDEX_TIP]
                cx, cy = smoothed_cursor.update(float(index_tip[0]), float(index_tip[1]))

                last_has_hand = True
                last_drawing_active = drawing_active
                last_cursor = [int(round(cx)), int(round(cy))]
            else:
                last_has_hand = False
                last_drawing_active = False
                last_cursor = None
                smoothed_cursor.valid = False

            next_tracking_at = now + tracking_interval

            emit({
                "type": "frame",
                "has_hand": last_has_hand,
                "drawing_active": last_drawing_active,
                "frame_size": [frame.shape[1], frame.shape[0]],
                "cursor": last_cursor,
            })

            if now - stats_last_emit >= 10.0:
                stats_last_emit = now
                with detection_lock:
                    stats = detection_stats.copy()
                emit({
                    "type": "status",
                    "level": "info",
                    "message": f"detector: processed={stats['processed_frames']} dropped={stats['dropped_frames']} latency={stats['last_latency_ms']:.1f}ms",
                })

    except KeyboardInterrupt:
        emit({"type": "status", "level": "info", "message": "Hand tracker stopped"})
    finally:
        stop_event.set()
        producer_thread.join(timeout=1.0)
        detector_thread.join(timeout=1.0)
        landmarker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
