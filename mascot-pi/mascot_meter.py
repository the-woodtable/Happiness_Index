#!/usr/bin/env python3
"""
Mascot engagement + happiness meter (multi-person), Raspberry Pi edition.

Reads the Pi Camera Module 3 (picamera2) by default, or an ESP32-S3 MJPEG stream / USB webcam,
tracks every visible face with a persistent ID, and measures engagement and happiness for each
person individually. Runs headless; --display shows the original OpenCV panel.

Metric definitions are unchanged from smile_meter_engagement.py (see that file's notes):
  ENGAGEMENT  stop-and-stay count, median eye contact time, median reaction time
  HAPPINESS   smiles, laughs, median time to happiness, average happiness, median duration
  SHOCK       scared / surprised count

What is new on the Pi:
  * Stats reset at 04:00 Singapore time every day (replaces the r key). Days are kept.
  * Every finished person is appended to data/people/people_<day>.jsonl (fsync'd), and today's
    totals and graph are rebuilt from it on start, so a power cut does not zero the day.
  * The numbers are uploaded to Firebase by mascot_cloud.py (numbers only, never images).
  * Raw blendshape CSVs are one file per day, gzipped after the day ends, and capped in size.
  * Waits for an NTP-synced clock before recording anything (the Pi has no RTC battery).
  * systemd notify + watchdog support.

Usage (on the Pi, inside the venv):
    python mascot_meter.py                         # Pi camera, 640x480, 12 fps, headless
    python mascot_meter.py --tune                  # also print face size / angles every second
    python mascot_meter.py --size 640x360 --fps 15
    python mascot_meter.py --source http://192.168.1.50:81/stream   # ESP32 (or --ip 192.168.1.50)
    python mascot_meter.py --source 0              # USB webcam
    python mascot_meter.py --display               # local window (needs a desktop + full OpenCV)

Notes on identity: IDs (P1, P2, ...) come from frame-to-frame position tracking, not face
recognition. If someone leaves for longer than TRACK_LOST_SECONDS and comes back, they get a new ID.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "face_landmarker.task")
SCHEMA_VERSION = 1
SGT = timezone(timedelta(hours=8))   # Singapore has no daylight saving
DAY_START_HOUR = 4                   # a new day starts at 04:00 SGT

# ---- tunables ---------------------------------------------------------------
SMOOTH_FRAMES = 5            # moving-average window applied to each channel
CALIB_SECONDS = 1            # baseline collection time, counted from the first look
MIN_CALIB_SAMPLES = 5
END_GAP_SECONDS = 2.0        # looked away this long DURING calibration -> calibration restarts

# looking at the mascot. These were tuned for the ESP32 view and WILL need retuning for the Pi
# camera: use --tune and the raw CSV, then set --min-face-frac / --yaw-max / --pitch-max.
YAW_MAX_DEG = 25.0
PITCH_MAX_DEG = 25.0
MIN_FACE_HEIGHT_FRAC = 0.18  # face box height / frame height (distance proxy)

# engagement
STAY_SECONDS = 5.0           # looked, then stayed in view this long (overridable with --stay-seconds)
MIN_TRACKED_SECONDS = 1.0    # tracks shorter than this are ignored (detector ghosts, fly-bys)

# smile
SMILE_DELTA_THRESHOLD = 0.25  # smile channel above own baseline by this much = smiling
SMILE_RELEASE_RATIO = 0.53     # hysteresis: smile "off" below threshold * this

# reaction: brow and mouth corners only
REACTION_CHANNELS = ("brow_up", "brow_down", "smile", "frown", "dimple")
REACTION_DELTA_THRESHOLD = 0.12  # any one of these rising this far above its own baseline
REACTION_HOLD_SECONDS = 0.20     # ...and staying there this long (rejects blink/jitter)

# laugh: weighted blend of channel rises above own baseline (no eyeLook* channels)
LAUGH_WEIGHTS = {"smile": 0.35, "jaw": 0.25, "cheek": 0.20, "eyesq": 0.20}
LAUGH_THRESHOLD = 0.15          # weighted score needed
LAUGH_MIN_SMILE_DELTA = 0.15    # gate: a laugh must also be a smile ...
LAUGH_MIN_JAW_DELTA = 0.05      # ... with the mouth open a little
LAUGH_HOLD_SECONDS = 0.20       # must be sustained this long to count

# shock: scared / surprised (combined)
SHOCK_HOLD_SECONDS = 0.3
SHOCK_SPECS = {
    "shocked": dict(
        weights={"brow_up": 0.5, "jaw": 0.5},
        threshold=0.06, gates={"brow_up": 0.15, "jaw": 0.1}, max_smile=None),
}
SHOCK_ORDER = ("shocked",)

# tracking
MATCH_MAX_DIST = 1.0         # max centre movement between frames, in units of the face's width
TRACK_LOST_SECONDS = 3.0     # a face unseen for this long is released

UPLIFT_FULL = 0.60           # smile delta that fills the on-screen bar
MAX_DT = 0.25                # per-frame time step cap (durations undercount below ~4 fps)

# operations
SUMMARY_EVERY_S = 0.5        # recompute the headline numbers this often
NO_FRAME_STATUS_S = 5.0      # no new frame for this long -> status "no_frames"
NO_FRAME_EXIT_S = 120.0      # ... for this long -> exit so systemd restarts us (fresh camera)
DISK_CHECK_EVERY_S = 60.0

IDLE, CALIBRATING, ACTIVE = "IDLE", "CALIBRATING", "ACTIVE"
STATE_COLOR = {IDLE: (140, 140, 140), CALIBRATING: (0, 220, 255), ACTIVE: (0, 200, 0)}  # BGR
PALETTE = [  # one colour per person (BGR), cycled by ID
    (255, 160, 60), (80, 200, 255), (120, 255, 120), (255, 120, 220),
    (60, 140, 255), (255, 230, 80), (200, 120, 255), (140, 255, 255),
]
# UI palette (BGR): #ffffff, #4f3a97, #fed200
WHITE = (255, 255, 255)
PURPLE = (151, 58, 79)
YELLOW = (0, 210, 254)

PANEL_W = 400
DIVIDER_W = 4
PANEL_MIN_H = 1016
MARGIN = 18
OVERLAY_REF_WIDTH = 1024.0

# ---- blendshapes ------------------------------------------------------------
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

CHANNEL_DEFS = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "cheek": ("cheekSquintLeft", "cheekSquintRight"),
    "eyesq": ("eyeSquintLeft", "eyeSquintRight"),
    "jaw": ("jawOpen",),
    "brow_up": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "brow_down": ("browDownLeft", "browDownRight"),
    "frown": ("mouthFrownLeft", "mouthFrownRight"),
    "dimple": ("mouthDimpleLeft", "mouthDimpleRight"),
}


def channels_from(blend: dict) -> dict:
    return {k: sum(blend.get(n, 0.0) for n in names) / len(names) for k, names in CHANNEL_DEFS.items()}


def log(msg: str):
    print(msg, flush=True)


# ---- time / days ------------------------------------------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def mono_to_ms(t_mono: float) -> int:
    """Convert a time.monotonic() value to epoch milliseconds (clock must already be synced)."""
    return int((time.time() - time.monotonic() + t_mono) * 1000)


def day_key(ms: int) -> str:
    """Singapore date of (time - 4 h): the 'business day' that starts at 04:00."""
    return (datetime.fromtimestamp(ms / 1000, SGT) - timedelta(hours=DAY_START_HOUR)).date().isoformat()


def day_start_ms(day: str) -> int:
    y, m, d = (int(x) for x in day.split("-"))
    return int(datetime(y, m, d, DAY_START_HOUR, tzinfo=SGT).timestamp() * 1000)


def clock_synced() -> bool:
    if os.path.exists("/run/systemd/timesync/synchronized"):
        return True
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out == "yes"
    except Exception:
        return False


# ---- systemd ----------------------------------------------------------------
def sd_notify(msg: str):
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
    except OSError:
        pass


class Watchdog:
    def __init__(self):
        self.last = 0.0

    def pet(self):
        t = time.monotonic()
        if t - self.last >= 5.0:
            sd_notify("WATCHDOG=1")
            self.last = t


# ---- safe files -------------------------------------------------------------
def atomic_write_json(path: str, obj):
    """Write-to-temp, fsync, rename: the file is either the old or the new version after a power cut."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    except OSError:
        pass


def read_jsonl(path: str) -> list:
    """Read a JSON-lines file, skipping a torn last line or any corrupt line (power cut)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
    return out


# ---- frame input ------------------------------------------------------------
class FrameGrabber(threading.Thread):
    """OpenCV source (ESP32 MJPEG URL or USB webcam). Keeps only the newest frame."""

    MAX_READ_FAILS = 30

    def __init__(self, source, size=None, fps=None):
        super().__init__(daemon=True)
        self.source = source
        self.size, self.fps = size, fps
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._halt = threading.Event()

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
            if self.size:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            if self.fps:
                cap.set(cv2.CAP_PROP_FPS, self.fps)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self):
        cap = None
        fails = 0
        while not self._halt.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open()
                if not cap.isOpened():
                    log(f"[grabber] cannot open {self.source!r}, retrying in 2s...")
                    time.sleep(2)
                    continue
                log(f"[grabber] connected to {self.source!r}")
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails < self.MAX_READ_FAILS:
                    time.sleep(0.02)
                    continue
                log(f"[grabber] {fails} failed reads in a row, reconnecting...")
                cap.release()
                cap = None
                fails = 0
                time.sleep(0.3)
                continue
            fails = 0
            with self._lock:
                self._frame = frame
                self._seq += 1
        if cap is not None:
            cap.release()

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def stop(self):
        self._halt.set()


class PiCamGrabber(threading.Thread):
    """Pi Camera Module 3 via picamera2. Delivers BGR frames (what OpenCV and the rest expect)."""

    def __init__(self, size, fps, full_fov=True, rotate=0):
        super().__init__(daemon=True)
        self.size, self.fps, self.full_fov, self.rotate = size, fps, full_fov, rotate
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._halt = threading.Event()

    def _start_camera(self):
        from picamera2 import Picamera2
        try:
            from libcamera import Transform, controls
        except ImportError:
            Transform = controls = None
        cam = Picamera2()
        frame_us = int(1_000_000 / self.fps)
        ctrl = {"FrameDurationLimits": (frame_us, frame_us)}
        if controls is not None:
            ctrl["AfMode"] = controls.AfModeEnum.Continuous   # Camera Module 3 has autofocus
        kw = dict(
            # picamera2's "RGB888" is B,G,R byte order in memory = OpenCV's BGR. Deliberate.
            main={"size": tuple(self.size), "format": "RGB888"},
            buffer_count=4,
            controls=ctrl,
        )
        if Transform is not None and self.rotate == 180:
            kw["transform"] = Transform(hflip=1, vflip=1)
        config = None
        if self.full_fov:
            # Without this, small outputs use the 1536x864 sensor mode, which is a CENTRE CROP of
            # the sensor (narrower view). 2304x1296 is the full sensor, 2x2 binned.
            for extra in ({"sensor": {"output_size": (2304, 1296)}}, {"raw": {"size": (2304, 1296)}}):
                try:
                    config = cam.create_video_configuration(**kw, **extra)
                    break
                except Exception:
                    config = None
        if config is None:
            config = cam.create_video_configuration(**kw)
        cam.configure(config)
        cam.start()
        log(f"[camera] started {self.size[0]}x{self.size[1]} @ {self.fps} fps, "
            f"sensor mode {config.get('sensor', {}) or config.get('raw', {}).get('size')}")
        return cam

    def run(self):
        cam = None
        fails = 0
        while not self._halt.is_set():
            try:
                if cam is None:
                    cam = self._start_camera()
                frame = cam.capture_array("main")
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                fails = 0
            except Exception as e:
                fails += 1
                log(f"[camera] error: {e!r} (attempt {fails})")
                try:
                    if cam is not None:
                        cam.stop()
                        cam.close()
                except Exception:
                    pass
                cam = None
                time.sleep(min(10, 2 * fails))
        if cam is not None:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def stop(self):
        self._halt.set()


# ---- face analysis ----------------------------------------------------------
def ensure_model(path: str) -> str:
    """The model is bundled; nothing is downloaded at the venue."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    sys.exit(f"Model file missing or broken: {path}\n"
             "Copy face_landmarker.task next to mascot_meter.py (see the setup guide).")


def head_pose_deg(matrix) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from MediaPipe's 4x4 facial transformation matrix."""
    R = np.array(matrix, dtype=float)[:3, :3]
    R = R / np.maximum(np.linalg.norm(R, axis=0), 1e-9)
    sy = math.hypot(R[0, 0], R[1, 0])
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return yaw, pitch, roll


@dataclass
class Face:
    box: tuple
    blend: dict
    yaw: float
    pitch: float
    roll: float
    height_frac: float


class FaceAnalyzer:
    def __init__(self, model_path: str, max_faces: int = 5):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = 0

    def analyze(self, bgr) -> list[Face]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return []
        faces: list[Face] = []
        blend = res.face_blendshapes or []
        mats = res.facial_transformation_matrixes or []
        for i, lms in enumerate(res.face_landmarks):
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            box = (int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h))
            scores = {c.category_name: c.score for c in blend[i]} if i < len(blend) else {}
            yaw = pitch = roll = 0.0
            if i < len(mats):
                yaw, pitch, roll = head_pose_deg(mats[i])
            faces.append(Face(box, scores, yaw, pitch, roll, max(ys) - min(ys)))
        return faces

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


# ---- per-person logic (unchanged metric logic) -------------------------------
def median_or_none(xs):
    return statistics.median(xs) if xs else None


@dataclass
class Person:
    """One tracked face: smoothing buffer, baseline, and every per-person metric."""
    pid: int
    box: tuple
    first_seen: float
    last_seen: float
    tracked: bool = True
    ch_buf: deque = field(default_factory=lambda: deque(maxlen=SMOOTH_FRAMES))

    state: str = IDLE
    calib: list = field(default_factory=list)
    calib_start: float = 0.0
    ref_time: Optional[float] = None
    baseline: Optional[dict] = None
    last_engaged: float = 0.0

    first_look: Optional[float] = None
    eye_time: float = 0.0
    stayed: bool = False

    react_time: Optional[float] = None
    react_channel: str = ""
    react_start: Optional[float] = None
    react_cand: str = ""

    smiling: bool = False
    tts: Optional[float] = None
    smile_time: float = 0.0
    streak: float = 0.0
    longest_smile: float = 0.0

    laughed: bool = False
    laugh_tts: Optional[float] = None
    laugh_time: float = 0.0
    laugh_start: Optional[float] = None
    peak_laugh: float = 0.0

    shock_seen: dict = field(default_factory=lambda: {e: False for e in SHOCK_ORDER})
    shock_tts: dict = field(default_factory=lambda: {e: None for e in SHOCK_ORDER})
    shock_time: dict = field(default_factory=lambda: {e: 0.0 for e in SHOCK_ORDER})
    shock_start: dict = field(default_factory=lambda: {e: None for e in SHOCK_ORDER})
    shock_peak: dict = field(default_factory=lambda: {e: 0.0 for e in SHOCK_ORDER})

    peaks: dict = field(default_factory=lambda: {k: 0.0 for k in CHANNEL_DEFS})

    # wall-clock moments (epoch ms) the headline flags first became true; used to rebuild the graph
    stayed_ms: Optional[int] = None
    smiled_ms: Optional[int] = None
    laughed_ms: Optional[int] = None

    face: Optional[Face] = None
    engaged: bool = False
    smooth: Optional[dict] = None
    delta: Optional[dict] = None
    smile_d: Optional[float] = None
    laugh_score: Optional[float] = None
    react_score: Optional[float] = None
    laughing: bool = False
    shock_score: Optional[dict] = None

    @property
    def happiness(self) -> Optional[float]:
        if not self.smiled or self.baseline is None:
            return None
        room_smile = max(1.0 - self.baseline["smile"], 1e-3)
        room_laugh = max(sum(w * (1.0 - self.baseline[k]) for k, w in LAUGH_WEIGHTS.items()), 1e-3)
        return min(1.0, max(self.peaks["smile"] / room_smile, self.peak_laugh / room_laugh))

    @property
    def color(self):
        return PALETTE[(self.pid - 1) % len(PALETTE)]

    @property
    def tracked_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def counted(self) -> bool:
        return self.tracked_seconds >= MIN_TRACKED_SECONDS

    @property
    def smiled(self) -> bool:
        return self.tts is not None

    def stamp_events(self, ms: int):
        if self.stayed and self.stayed_ms is None:
            self.stayed_ms = ms
        if self.smiled and self.smiled_ms is None:
            self.smiled_ms = ms
        if self.laughed and self.laughed_ms is None:
            self.laughed_ms = ms

    def release(self):
        self.tracked = False
        self.face = None
        self.engaged = False
        self.smooth = self.delta = None
        self.smile_d = self.laugh_score = self.react_score = None
        self.laughing = False
        self.shock_score = None

    def step(self, face: Optional[Face], now: float, dt: float):
        self.face = face
        self.engaged = (
            face is not None
            and abs(face.yaw) <= YAW_MAX_DEG
            and abs(face.pitch) <= PITCH_MAX_DEG
            and face.height_frac >= MIN_FACE_HEIGHT_FRAC
        )
        self.smile_d = self.laugh_score = self.react_score = None
        self.laughing = False
        self.shock_score = None
        self.delta = None
        self.smooth = None

        if face is not None:
            self.ch_buf.append(channels_from(face.blend))
            n = len(self.ch_buf)
            self.smooth = {k: sum(c[k] for c in self.ch_buf) / n for k in CHANNEL_DEFS}
            self.box = face.box
            self.last_seen = now
        else:
            self.ch_buf.clear()

        if self.engaged:
            self.last_engaged = now
            self.eye_time += dt
            if self.first_look is None:
                self.first_look = now
        if (self.first_look is not None and not self.stayed
                and self.last_seen - self.first_look >= STAY_SECONDS):
            self.stayed = True

        if self.state == IDLE:
            if self.engaged:
                self.state, self.calib, self.calib_start = CALIBRATING, [self.smooth], now
            return
        if self.state == CALIBRATING:
            if self.engaged:
                self.calib.append(self.smooth)
                if now - self.calib_start >= CALIB_SECONDS and len(self.calib) >= MIN_CALIB_SAMPLES:
                    self.baseline = {k: statistics.median(s[k] for s in self.calib) for k in CHANNEL_DEFS}
                    self.ref_time = self.calib_start
                    self.state = ACTIVE
            elif now - self.last_engaged >= END_GAP_SECONDS:
                self.state = IDLE
            return

        if not self.engaged:
            self.smiling = False
            self.streak = 0.0
            self.react_start = None
            self.laugh_start = None
            self.shock_start = {e: None for e in SHOCK_ORDER}
            return

        d = {k: self.smooth[k] - self.baseline[k] for k in CHANNEL_DEFS}
        self.delta = d
        for k, v in d.items():
            if v > self.peaks[k]:
                self.peaks[k] = v

        sd = d["smile"]
        self.smile_d = sd
        if self.smiling:
            self.smiling = sd >= SMILE_DELTA_THRESHOLD * SMILE_RELEASE_RATIO
        else:
            self.smiling = sd >= SMILE_DELTA_THRESHOLD
        if self.smiling:
            self.smile_time += dt
            self.streak += dt
            self.longest_smile = max(self.longest_smile, self.streak)
            if self.tts is None:
                self.tts = now - self.ref_time
        else:
            self.streak = 0.0

        ls = sum(w * max(d[k], 0.0) for k, w in LAUGH_WEIGHTS.items())
        self.laugh_score = ls
        self.peak_laugh = max(self.peak_laugh, ls)
        self.laughing = (ls >= LAUGH_THRESHOLD and d["smile"] >= LAUGH_MIN_SMILE_DELTA
                         and d["jaw"] >= LAUGH_MIN_JAW_DELTA)
        if self.laughing:
            self.laugh_time += dt
            if self.laugh_start is None:
                self.laugh_start = now
            if not self.laughed and now - self.laugh_start >= LAUGH_HOLD_SECONDS:
                self.laughed = True
                self.laugh_tts = self.laugh_start - self.ref_time
        else:
            self.laugh_start = None

        self.shock_score = {}
        for emo in SHOCK_ORDER:
            spec = SHOCK_SPECS[emo]
            sc = sum(w * max(d[k], 0.0) for k, w in spec["weights"].items())
            self.shock_score[emo] = sc
            self.shock_peak[emo] = max(self.shock_peak[emo], sc)
            on = (sc >= spec["threshold"]
                  and all(d[k] >= v for k, v in spec["gates"].items())
                  and (spec["max_smile"] is None or d["smile"] < spec["max_smile"]))
            if on:
                self.shock_time[emo] += dt
                if self.shock_start[emo] is None:
                    self.shock_start[emo] = now
                if not self.shock_seen[emo] and now - self.shock_start[emo] >= SHOCK_HOLD_SECONDS:
                    self.shock_seen[emo] = True
                    self.shock_tts[emo] = self.shock_start[emo] - self.ref_time
            else:
                self.shock_start[emo] = None

        best_ch, best = max(((k, d[k]) for k in REACTION_CHANNELS), key=lambda kv: kv[1])
        self.react_score = best
        if self.react_time is None:
            if best >= REACTION_DELTA_THRESHOLD:
                if self.react_start is None:
                    self.react_start, self.react_cand = now, best_ch
                if now - self.react_start >= REACTION_HOLD_SECONDS:
                    self.react_time = self.react_start - self.ref_time
                    self.react_channel = self.react_cand
            else:
                self.react_start = None


# ---- person records and day summary ------------------------------------------
# A "record" is the per-person dict that is stored locally (people_<day>.jsonl) and, minus the local
# fields, uploaded to /days/<day>/people/<key>. The day summary is computed from records only, so it
# can be rebuilt exactly after a restart.
def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def person_record(p: Person) -> dict:
    rec = {
        "v": SCHEMA_VERSION,
        "key": f"{mono_to_ms(p.first_seen) // 1000}-{p.pid}",
        "firstSeenMs": mono_to_ms(p.first_seen),
        "endMs": mono_to_ms(p.last_seen),
        "trackedS": _r(p.tracked_seconds),
        "eyeContactS": _r(p.eye_time),
        "stayed": bool(p.stayed),
        "reactionS": _r(p.react_time),
        "reactionChannel": p.react_channel if p.react_time is not None else None,
        "smiled": bool(p.smiled),
        "timeToSmileS": _r(p.tts),
        "smileTimeS": _r(p.smile_time) if p.smiled else None,
        "laughed": bool(p.laughed),
        "peakLaugh": _r(p.peak_laugh),
        "happiness": _r(p.happiness),
        "shocked": bool(p.shock_seen["shocked"]),
        "stayedMs": p.stayed_ms,
        "smiledMs": p.smiled_ms,
        "laughedMs": p.laughed_ms,
    }
    return {k: v for k, v in rec.items() if v is not None}


def summarise_records(recs: list) -> dict:
    """Same headline numbers as the original summarise(), computed from records of counted people."""
    looked = [r for r in recs if r.get("eyeContactS", 0) > 0]
    h = [r["happiness"] for r in recs if "happiness" in r]
    return {
        "n": len(recs),
        "stayed": sum(1 for r in recs if r.get("stayed")),
        "smiles": sum(1 for r in recs if r.get("smiled")),
        "laughs": sum(1 for r in recs if r.get("laughed")),
        "eye": median_or_none([r["eyeContactS"] for r in looked]),
        "reaction": median_or_none([r["reactionS"] for r in recs if "reactionS" in r]),
        "tts": median_or_none([r["timeToSmileS"] for r in recs if "timeToSmileS" in r]),
        "duration": median_or_none([r["smileTimeS"] for r in recs if r.get("smiled") and "smileTimeS" in r]),
        "happiness": statistics.mean(h) if h else None,
        "shocked": sum(1 for r in recs if r.get("shocked")),
    }


def total_happiness(s: dict) -> Optional[float]:
    """Total Happiness Index, 0..100 = (share of people who smiled) x (average happiness, 0..1) x 100.
    None until anyone is tracked; 0 if people were tracked but nobody has smiled yet."""
    if not s["n"]:
        return None
    if not s["smiles"] or s["happiness"] is None:
        return 0.0
    return 100.0 * (s["smiles"] / s["n"]) * s["happiness"]


def contract_summary(s: dict, day: str) -> dict:
    """Panel summary -> /live/summary payload (without updatedMs). Missing = no data yet."""
    out = {"v": SCHEMA_VERSION, "day": day, "n": s["n"], "stayed": s["stayed"], "smiles": s["smiles"],
           "laughs": s["laughs"], "shocked": s["shocked"], "stayThresholdS": STAY_SECONDS}
    for src, dst in (("eye", "eyeContactMedianS"), ("reaction", "reactionMedianS"),
                     ("tts", "timeToHappinessMedianS"), ("duration", "happinessDurationMedianS")):
        if s[src] is not None:
            out[dst] = round(s[src], 1)
    if s["happiness"] is not None:
        out["happiness"] = int(round(s["happiness"] * 100))
    t = total_happiness(s)
    if t is not None:
        out["totalHappiness"] = int(round(t))
    return out


def day_index_entry(s: dict) -> dict:
    return {k: s[k] for k in ("n", "stayed", "smiles", "laughs", "shocked")}


class DayTimeline:
    """Cumulative [t, engaged, smiled, laughed] rows, t = whole seconds since 04:00 SGT."""
    KEYS = ("engaged", "smiles", "laughs")

    def __init__(self, day: str, rows=None):
        self.day = day
        self.start_ms = day_start_ms(day)
        self.rows = rows or [[0, 0, 0, 0]]

    @classmethod
    def from_records(cls, day: str, recs: list) -> "DayTimeline":
        tl = cls(day)
        events = []
        for r in recs:
            for i, k in ((1, "stayedMs"), (2, "smiledMs"), (3, "laughedMs")):
                if k in r:
                    events.append((r[k], i))
        events.sort()
        counts = [0, 0, 0]
        for ms, i in events:
            counts[i - 1] += 1
            tl._put(ms, counts)
        return tl

    def _put(self, ms: int, counts):
        t = max(0, min(86400, int((ms - self.start_ms) / 1000)))
        row = [t] + list(counts)
        if self.rows[-1][1:] == row[1:]:
            return False
        if t <= self.rows[-1][0]:
            self.rows[-1] = [self.rows[-1][0]] + list(counts)
        else:
            self.rows.append(row)
        return True

    def update(self, s: dict, ms: int) -> bool:
        return self._put(ms, [s["stayed"], s["smiles"], s["laughs"]])

    # compatibility with the --display graph code
    @property
    def last(self) -> dict:
        r = self.rows[-1]
        return {"engaged": r[1], "smiles": r[2], "laughs": r[3]}

    @property
    def points(self) -> dict:
        return {k: [(r[0], r[i + 1]) for r in self.rows] for i, k in enumerate(self.KEYS)}

    def elapsed(self, _now=None) -> float:
        return max(0.0, (now_ms() - self.start_ms) / 1000)


# ---- tracking ----------------------------------------------------------------
class Tracker:
    """Associates detections with people across frames (nearest centre, greedy)."""

    def __init__(self):
        self.people: dict[int, Person] = {}
        self.next_id = 1

    def assign(self, faces: list[Face], now: float) -> dict[int, Face]:
        pairs = []
        for p in self.people.values():
            pcx, pcy = (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2
            pw = max(p.box[2] - p.box[0], 1)
            for fi, f in enumerate(faces):
                fcx, fcy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
                d = math.hypot(fcx - pcx, fcy - pcy) / pw
                if d <= MATCH_MAX_DIST:
                    pairs.append((d, p.pid, fi))
        pairs.sort()
        matched: dict[int, Face] = {}
        used_faces: set[int] = set()
        for _, pid, fi in pairs:
            if pid in matched or fi in used_faces:
                continue
            matched[pid] = faces[fi]
            used_faces.add(fi)
        for fi, f in enumerate(faces):
            if fi not in used_faces:
                p = Person(pid=self.next_id, box=f.box, first_seen=now, last_seen=now)
                self.next_id += 1
                self.people[p.pid] = p
                matched[p.pid] = f
        return matched

    def retire_lost(self, now: float) -> list[Person]:
        return [p for p in self.people.values() if now - p.last_seen > TRACK_LOST_SECONDS]

    def drop(self, p: Person):
        self.people.pop(p.pid, None)


# ---- local storage -----------------------------------------------------------
def open_csv(path: str, fields: list):
    """Open a CSV for appending. If it exists with a different header it is moved aside."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header == fields:
            f = open(path, "a", newline="")
            return f, csv.writer(f)
        backup = f"{path}.{time.strftime('%Y%m%d-%H%M%S')}.old"
        os.replace(path, backup)
        log(f"[log] {path} had a different format; moved to {backup}")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(fields)
    f.flush()
    return f, w


def _f(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


class RawLog:
    """Every frame, every face: all raw blendshapes + head pose + the live scores. One file per day."""
    FIELDS = (["run_id", "utc_ms", "wall_time_sgt", "elapsed_s", "frame", "person_id", "state", "engaged",
               "yaw", "pitch", "roll", "face_height_frac"]
              + BLENDSHAPE_NAMES
              + ["smile_delta", "laugh_score", "react_score", "smiling", "laughing"]
              + [f"{e}_score" for e in SHOCK_ORDER])

    def __init__(self, raw_dir: Optional[str], run_id: str, t0: float):
        self.dir = raw_dir
        self.run_id, self.t0 = run_id, t0
        self.f = self.w = None
        self.day = None
        self.paused = False
        self.last_flush = time.monotonic()

    @property
    def enabled(self):
        return bool(self.dir)

    def path_for(self, day):
        return os.path.join(self.dir, f"raw_{day}.csv")

    def open_day(self, day: str):
        if not self.enabled:
            return
        self.close()
        self.day = day
        self.f, self.w = open_csv(self.path_for(day), self.FIELDS)

    def write(self, frame_idx: int, now: float, p: Person):
        if not self.enabled or self.paused or self.w is None or p.face is None:
            return
        fc = p.face
        ms = mono_to_ms(now)
        row = [self.run_id, ms, datetime.fromtimestamp(ms / 1000, SGT).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
               f"{now - self.t0:.3f}", frame_idx, f"P{p.pid}", p.state, int(p.engaged),
               f"{fc.yaw:.1f}", f"{fc.pitch:.1f}", f"{fc.roll:.1f}", f"{fc.height_frac:.3f}"]
        row += [f"{fc.blend.get(n, 0.0):.4f}" for n in BLENDSHAPE_NAMES]
        row += [_f(p.smile_d, 4), _f(p.laugh_score, 4), _f(p.react_score, 4),
                int(p.smiling), int(p.laughing)]
        row += [_f(None if p.shock_score is None else p.shock_score[e], 4) for e in SHOCK_ORDER]
        self.w.writerow(row)
        if now - self.last_flush > 1.0:
            self.f.flush()
            self.last_flush = now

    def close(self):
        if self.f is not None:
            try:
                self.f.flush()
                self.f.close()
            except Exception:
                pass
        self.f = self.w = None


class PersonCSV:
    """One row per person (every tracked face, including uncounted), one file per day. Same columns
    as the original people_summary.csv plus the day and UTC times."""
    FIELDS = (["run_id", "day", "person_id", "counted", "first_seen_utc_ms", "first_seen_s", "tracked_s",
               "first_look_s", "eye_contact_s", "stayed", "stay_s", "reaction_time_s", "reaction_channel",
               "smile_tts_s", "smiled", "smile_time_s", "longest_smile_s",
               "laughed", "laugh_tts_s", "laugh_time_s", "peak_laugh_score", "happiness"]
              + [f"{e}_{c}" for e in SHOCK_ORDER for c in ("seen", "tts_s", "time_s", "peak_score")]
              + [f"baseline_{k}" for k in CHANNEL_DEFS]
              + [f"peak_{k}" for k in CHANNEL_DEFS])

    def __init__(self, people_dir: str, run_id: str, t0: float):
        self.dir, self.run_id, self.t0 = people_dir, run_id, t0
        self.f = self.w = None
        self.day = None

    def open_day(self, day: str):
        self.close()
        self.day = day
        self.f, self.w = open_csv(os.path.join(self.dir, f"people_{day}.csv"), self.FIELDS)

    def write(self, p: Person):
        stay = None if p.first_look is None else max(0.0, p.last_seen - p.first_look)
        row = [self.run_id, self.day, f"P{p.pid}", int(p.counted), mono_to_ms(p.first_seen),
               _f(p.first_seen - self.t0, 2), _f(p.tracked_seconds, 2),
               _f(None if p.first_look is None else p.first_look - self.t0, 2),
               _f(p.eye_time, 2), int(p.stayed), _f(stay, 2),
               _f(p.react_time, 2), p.react_channel,
               _f(p.tts, 2), int(p.smiled), _f(p.smile_time, 2), _f(p.longest_smile, 2),
               int(p.laughed), _f(p.laugh_tts, 2), _f(p.laugh_time, 2), _f(p.peak_laugh, 3),
               _f(p.happiness, 3)]
        for e in SHOCK_ORDER:
            row += [int(p.shock_seen[e]), _f(p.shock_tts[e], 2), _f(p.shock_time[e], 2),
                    _f(p.shock_peak[e], 3)]
        row += [_f(None if p.baseline is None else p.baseline[k], 4) for k in CHANNEL_DEFS]
        row += [_f(p.peaks[k], 4) for k in CHANNEL_DEFS]
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
        self.f = self.w = None


class Store:
    """data/
         people/people_<day>.jsonl   counted people (the source of truth for totals and uploads)
         people/people_<day>.csv     every person, full detail, for analysis
         raw/raw_<day>.csv(.gz)      every frame, size-capped
         days/<day>.json             final summary + graph of a finished day (uploaded by the cloud side)
         cloud_state.json            upload progress (written by mascot_cloud.py)"""

    def __init__(self, root: str):
        self.root = root
        self.people_dir = os.path.join(root, "people")
        self.raw_dir = os.path.join(root, "raw")
        self.days_dir = os.path.join(root, "days")
        for d in (self.people_dir, self.raw_dir, self.days_dir):
            os.makedirs(d, exist_ok=True)

    def jsonl_path(self, day):
        return os.path.join(self.people_dir, f"people_{day}.jsonl")

    def final_path(self, day):
        return os.path.join(self.days_dir, f"{day}.json")

    def load_records(self, day) -> list:
        self.repair_tail(day)
        return read_jsonl(self.jsonl_path(day))

    def repair_tail(self, day):
        """A power cut can leave half a line at the end. Cut it off, so the next record is not glued
        onto it."""
        path = self.jsonl_path(day)
        try:
            with open(path, "rb+") as f:
                data = f.read()
                if data and not data.endswith(b"\n"):
                    keep = data.rfind(b"\n") + 1
                    f.truncate(keep)
                    f.flush()
                    os.fsync(f.fileno())
                    log(f"[disk] removed a half-written line at the end of {os.path.basename(path)}")
        except FileNotFoundError:
            pass

    def append_record(self, day, rec):
        with open(self.jsonl_path(day), "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_final(self, day, recs):
        s = summarise_records(recs)
        summary = contract_summary(s, day)
        summary["updatedMs"] = int(max([r.get("endMs", 0) for r in recs] + [day_start_ms(day)]))
        atomic_write_json(self.final_path(day), {
            "day": day,
            "summary": summary,
            "timeline": {"points": DayTimeline.from_records(day, recs).rows},
            "index": day_index_entry(s),
        })

    def finalise_missing(self, today: str, keep_days=30):
        """Write days/<day>.json for any past day that has people but no final file (e.g. the Pi was
        off at 04:00)."""
        for name in sorted(os.listdir(self.people_dir)):
            if not (name.startswith("people_") and name.endswith(".jsonl")):
                continue
            day = name[len("people_"):-len(".jsonl")]
            if day >= today or os.path.exists(self.final_path(day)):
                continue
            try:
                age = (datetime.fromisoformat(today) - datetime.fromisoformat(day)).days
            except ValueError:
                continue
            if age <= keep_days:
                self.write_final(day, self.load_records(day))
                log(f"[day] wrote final totals for {day}")


def compress_old_raw(raw_dir: str, today: str):
    """Gzip finished days' raw CSVs (about 5x smaller). Runs in a background thread."""
    for name in sorted(os.listdir(raw_dir)):
        if not (name.startswith("raw_") and name.endswith(".csv")):
            continue
        if name[4:14] >= today:
            continue
        src = os.path.join(raw_dir, name)
        tmp = src + ".gz.tmp"
        try:
            with open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            os.replace(tmp, src + ".gz")
            os.remove(src)
            log(f"[disk] compressed {name}")
        except Exception as e:
            log(f"[disk] could not compress {name}: {e!r}")
            try:
                os.remove(tmp)
            except OSError:
                pass


def enforce_disk(raw_dir: str, current: Optional[str], max_bytes: int, min_free: int) -> bool:
    """Delete the oldest raw logs until under the cap and enough free space remains.
    Returns False if raw logging must pause (card nearly full even after deleting old raw logs)."""
    files = []
    for name in os.listdir(raw_dir):
        if name.startswith("raw_") and (name.endswith(".csv") or name.endswith(".csv.gz")):
            path = os.path.join(raw_dir, name)
            try:
                files.append((name, path, os.path.getsize(path)))
            except OSError:
                pass
    files.sort()
    total = sum(sz for _, _, sz in files)
    for name, path, sz in files:
        free = shutil.disk_usage(raw_dir).free
        if total <= max_bytes and free >= min_free:
            break
        if path == current:
            continue
        try:
            os.remove(path)
            total -= sz
            log(f"[disk] deleted old raw log {name} ({sz / 1e6:.0f} MB) to stay under the cap")
        except OSError:
            pass
    return shutil.disk_usage(raw_dir).free >= min_free and total <= max_bytes * 1.05


# ---- drawing (only used with --display; unchanged apart from the footer) ------
SS = 3
PANEL_REFRESH_HZ = 2
PALE_PURPLE = (236, 228, 244)
CORAL = (91, 84, 242)
YELLOW_EDGE = (0, 150, 190)
CORAL_EDGE = (50, 40, 170)

_FONT_FILES = {
    False: ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "LiberationSans-Regular.ttf", "FreeSans.ttf", "segoeui.ttf", "arial.ttf"],
    True: ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "LiberationSans-Bold.ttf", "FreeSansBold.ttf", "segoeuib.ttf", "arialbd.ttf"],
}


@lru_cache(maxsize=256)
def _font(px: int, bold: bool):
    for name in _FONT_FILES[bold]:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(px)
    except TypeError:
        return ImageFont.load_default()


def get_font(px: float, bold: bool = False):
    return _font(max(6, int(round(px))), bool(bold))


def _rgb(c):
    return (c[2], c[1], c[0])


_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


class TextLayer:
    def __init__(self):
        self.items = []

    def add(self, x, y, text, px, color, bold=False, outline=None):
        self.items.append((int(x), int(y), text, px, color, bold, outline))

    def apply(self, frame):
        if not self.items:
            return frame
        img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        d = ImageDraw.Draw(img)
        for x, y, text, px, color, bold, outline in self.items:
            f = get_font(px, bold)
            if outline is not None:
                d.text((x, y), text, font=f, fill=_rgb(color), anchor="ls",
                       stroke_width=max(1, int(round(px / 9))), stroke_fill=_rgb(outline))
            else:
                d.text((x, y), text, font=f, fill=_rgb(color), anchor="ls")
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


class Canvas:
    def __init__(self, w, h, s=1.0, bg=WHITE):
        self.bw, self.bh, self.s = w, h, s
        self.W, self.H = int(round(w * s)), int(round(h * s))
        self.k = s * SS
        self.big = Image.new("RGB", (self.W * SS, self.H * SS), _rgb(bg))
        self.d = ImageDraw.Draw(self.big)
        self.texts = []

    def _pt(self, x, y):
        return (int(round(x * self.k)), int(round(y * self.k)))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0):
        a, b = self._pt(x0, y0), self._pt(x1, y1)
        self.d.rectangle([a, (b[0] - 1, b[1] - 1)],
                         fill=None if fill is None else _rgb(fill),
                         outline=None if outline is None else _rgb(outline),
                         width=max(1, int(round(width * self.k))))

    def line(self, pts, color, width=1.0):
        self.d.line([self._pt(x, y) for x, y in pts], fill=_rgb(color),
                    width=max(1, int(round(width * self.k))), joint="curve")

    def circle(self, x, y, r, fill=None, outline=None, width=1.0):
        a, b = self._pt(x - r, y - r), self._pt(x + r, y + r)
        self.d.ellipse([a, b], fill=None if fill is None else _rgb(fill),
                       outline=None if outline is None else _rgb(outline),
                       width=max(1, int(round(width * self.k))))

    def text(self, x, y, text, size, color, bold=False, anchor="l"):
        self.texts.append((x, y, text, size, color, bold, anchor))

    def text_w(self, text, size, bold=False):
        return get_font(size * self.s, bold).getlength(text) / self.s

    def finish(self):
        img = self.big.resize((self.W, self.H), _LANCZOS)
        d = ImageDraw.Draw(img)
        for x, y, text, size, color, bold, anchor in self.texts:
            d.text((x * self.s, y * self.s), text, font=get_font(size * self.s, bold),
                   fill=_rgb(color), anchor={"l": "ls", "m": "ms", "r": "rs"}[anchor])
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def draw_person(frame, layer: TextLayer, p: Person, S: float, k: float, vid_h: float):
    if p.face is None:
        return
    x1, y1, x2, y2 = (int(v * S) for v in p.face.box)
    col = p.color if p.engaged else STATE_COLOR[IDLE]
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(2, int(round(2 * k))))
    status = p.state if p.engaged else "away"
    tag = " LAUGH" if p.laughing else (" STAY" if p.stayed else "")
    layer.add(x1, max(y1 - int(8 * k), int(20 * k)), f"P{p.pid} {status}{tag}", 20 * k, col, True, PURPLE)
    bw, bh = max(x2 - x1, int(50 * k)), int(10 * k)
    by = y2 + int(8 * k)
    if by + bh + int(30 * k) < vid_h:
        cv2.rectangle(frame, (x1, by), (x1 + bw, by + bh), PURPLE, -1)
        if p.smile_d is not None:
            fill = int(bw * min(max(p.smile_d / UPLIFT_FULL, 0.0), 1.0))
            cv2.rectangle(frame, (x1, by), (x1 + fill, by + bh), YELLOW, -1)
        tx = x1 + int(bw * SMILE_DELTA_THRESHOLD / UPLIFT_FULL)
        cv2.line(frame, (tx, by - 2), (tx, by + bh + 2), WHITE, 2)
        cv2.rectangle(frame, (x1, by), (x1 + bw, by + bh), WHITE, 1)
        s_txt = "--" if p.smile_d is None else f"{p.smile_d:+.2f}"
        l_txt = "--" if p.laugh_score is None else f"{p.laugh_score:.2f}"
        layer.add(x1, by + bh + int(18 * k), f"smile {s_txt}  laugh {l_txt}", 16 * k, WHITE, False, PURPLE)


def draw_overlay(layer: TextLayer, n_faces, fps, k: float, vid_h: float):
    layer.add(int(14 * k), int(34 * k), f"Faces: {n_faces}", 28 * k, WHITE, True, PURPLE)
    layer.add(int(14 * k), int(vid_h - 14 * k), f"FPS {fps:.1f}", 17 * k, WHITE, False, PURPLE)


SERIES = (
    ("engaged", "Engaged", PURPLE, None),
    ("smiles", "Smiled", YELLOW, YELLOW_EDGE),
    ("laughs", "Laughed", CORAL, CORAL_EDGE),
)


def _nice_step(span: float, candidates, max_ticks: int) -> float:
    for c in candidates:
        if span / c <= max_ticks:
            return c
    return candidates[-1]


def _fmt_clock(sec: float) -> str:
    mins = int(round((DAY_START_HOUR * 3600 + sec) / 60)) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"


def draw_graph(cv: Canvas, x0, y0, w, h, tl: DayTimeline, now: float):
    cv.text(x0, y0 + 13, "Number of people over time", 14, PURPLE, True)
    lx, ly = x0, y0 + 33
    for key, label, color, edge in SERIES:
        txt = f"{label} {tl.last[key]}"
        if edge:
            cv.line([(lx, ly - 4), (lx + 16, ly - 4)], edge, 5)
        cv.line([(lx, ly - 4), (lx + 16, ly - 4)], color, 3)
        cv.text(lx + 21, ly, txt, 12, PURPLE)
        lx += 21 + cv.text_w(txt, 12) + 16

    left, right = x0 + 28, x0 + w - 6
    top, bottom = y0 + 46, y0 + h - 20
    pw, ph = right - left, bottom - top
    if pw < 40 or ph < 40:
        return
    elapsed = tl.elapsed(now)
    top_count = max(tl.last.values())
    x_candidates = [600, 900, 1800, 3600, 7200, 10800, 14400, 21600]
    xt = _nice_step(max(elapsed, 3600.0), x_candidates, 5)
    xmax = max(xt, math.ceil(max(elapsed, 3600.0) / xt) * xt)
    y_candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    yt = _nice_step(max(top_count, 4), y_candidates, 5)
    ymax = max(yt * 4, math.ceil(max(top_count, 4) / yt) * yt)

    def X(t):
        return left + pw * t / xmax

    def Y(c):
        return bottom - ph * c / ymax

    cv.rect(left, top, right, bottom, fill=PALE_PURPLE)
    c = 0
    while c <= ymax + 1e-9:
        y = Y(c)
        cv.line([(left, y), (right, y)], WHITE, 1)
        cv.text(left - 5, y + 4, str(int(c)), 11, PURPLE, anchor="r")
        c += yt
    t = 0.0
    while t <= xmax + 1e-9:
        x = X(t)
        cv.line([(x, top), (x, bottom)], WHITE, 1)
        cv.text(x, bottom + 14, _fmt_clock(t), 11, PURPLE, anchor="m")
        t += xt
    pts_all = tl.points
    for key, label, color, edge in SERIES:
        pts = pts_all[key] + [(elapsed, tl.last[key])]
        poly = [(X(pts[0][0]), Y(pts[0][1]))]
        for (_, c_prev), (t_next, c_next) in zip(pts, pts[1:]):
            poly.append((X(t_next), Y(c_prev)))
            poly.append((X(t_next), Y(c_next)))
        if edge:
            cv.line(poly, edge, 5)
        cv.line(poly, color, 3)
        cv.circle(X(elapsed), Y(tl.last[key]), 4, fill=color, outline=PURPLE, width=1)
    cv.line([(left, top), (left, bottom)], PURPLE, 2)
    cv.line([(left, bottom), (right, bottom)], PURPLE, 2)


def fmt_sec(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.1f} sec"


def draw_stat(cv: Canvas, x, y, w, n: int, total: int, size: float):
    pct = "(--)" if total == 0 else f"({100.0 * n / total:.0f}%)"
    ns, ts = str(n), f"/ {total}"
    while True:
        wn = cv.text_w(ns, size, True)
        wt = cv.text_w(ts, size * 0.45)
        wp = cv.text_w(pct, size * 0.65)
        if wn + 6 + wt + 14 + wp <= w or size <= 14:
            break
        size -= 1
    cv.text(x, y, ns, size, PURPLE, True)
    cv.text(x + wn + 6, y, ts, size * 0.45, PURPLE)
    cv.text(x + w, y, pct, size * 0.65, PURPLE, anchor="r")


def _wrap2(cv: Canvas, text: str, size: float, w: float) -> list:
    if cv.text_w(text, size) <= w:
        return [text]
    words = text.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = max(cv.text_w(a, size), cv.text_w(b, size))
        if best is None or score < best[0]:
            best = (score, [a, b])
    return best[1] if best else [text]


def draw_median(cv: Canvas, x, y, w, label: str, value: str):
    for i, line in enumerate(_wrap2(cv, label, 13, w)):
        cv.text(x, y + 15 * i, line, 13, PURPLE)
    cv.text(x, y + 47, value, 24, PURPLE)


def draw_panel(cv: Canvas, s: dict, tl: DayTimeline, now: float):
    m = MARGIN
    inner_w = PANEL_W - 2 * m
    gap = 24
    col_w = (inner_w - gap) / 2
    x2 = m + col_w + gap

    def header(y, title):
        cv.rect(0, y, PANEL_W, y + 34, fill=PURPLE)
        cv.text(m, y + 23, title, 17, WHITE, True)

    def divider(y):
        cv.line([(m, y), (PANEL_W - m, y)], YELLOW, 3)

    header(0, "ENGAGEMENT")
    cv.text(m, 62, "Number of people engaged", 16, PURPLE, True)
    cv.text(m, 82, f"Stopped to look for {STAY_SECONDS:g} seconds +", 13, PURPLE)
    draw_stat(cv, m, 144, inner_w, s["stayed"], s["n"], 66)
    divider(160)
    draw_median(cv, m, 184, col_w, "Median eye contact time", fmt_sec(s["eye"]))
    draw_median(cv, x2, 184, col_w, "Median reaction time", fmt_sec(s["reaction"]))

    header(248, "HAPPINESS")
    cv.text(m, 308, "Number of smiles", 15, PURPLE, True)
    cv.text(x2, 308, "Number of laughs", 15, PURPLE, True)
    draw_stat(cv, m, 354, col_w, s["smiles"], s["n"], 46)
    draw_stat(cv, x2, 354, col_w, s["laughs"], s["n"], 46)
    divider(372)
    draw_median(cv, m, 396, col_w, "Median time to happiness", fmt_sec(s["tts"]))
    draw_median(cv, x2, 396, col_w, "Median duration of happiness", fmt_sec(s["duration"]))
    divider(458)
    cv.text(m, 489, "Average happiness (out of 100)", 13, PURPLE)
    hv = s["happiness"]
    cv.text(PANEL_W - m, 492, "--" if hv is None else f"{hv * 100:.0f}", 26, PURPLE, True, anchor="r")
    divider(506)
    cv.text(m, 534, "Total Happiness Index", 16, PURPLE, True)
    cv.text(m, 552, "% of people smiled * average happiness, out of 100", 11, PURPLE)
    tv = total_happiness(s)
    cv.text(m, 626, "--" if tv is None else f"{tv:.0f}", 76, PURPLE, True)
    divider(642)

    header(650, "SHOCK")
    cv.text(m, 708, "Number of people scared / surprised", 16, PURPLE, True)
    draw_stat(cv, m, 758, inner_w, s["shocked"], s["n"], 46)
    divider(772)

    g_top = 784
    draw_graph(cv, m, g_top, inner_w, cv.bh - 26 - g_top, tl, now)
    cv.text(m, cv.bh - 8, "q quit   (stats reset daily at 04:00)", 13, PURPLE)


def compose(frame, panel, S: float = 1.0):
    h = panel.shape[0]
    divider = np.full((h, max(1, int(round(DIVIDER_W * S))), 3), YELLOW, np.uint8)
    return np.hstack([frame, divider, panel])


class Display:
    """The original GUI, unchanged in look. Only created with --display."""

    def __init__(self, scale: float):
        self.S = scale
        self.panel_img = None
        self.panel_key = None

    def show(self, frame, active, n_faces, fps, summary, timeline, now) -> bool:
        """Returns False if the user pressed q."""
        fh, fw = frame.shape[:2]
        S = self.S
        H = max(fh, PANEL_MIN_H)
        if fh < H:
            frame = np.vstack([frame, np.full((H - fh, fw, 3), WHITE, np.uint8)])
        if S != 1.0:
            frame = cv2.resize(frame, (int(round(fw * S)), int(round(H * S))),
                               interpolation=cv2.INTER_AREA if S < 1.0 else cv2.INTER_LINEAR)
        k = (fw / OVERLAY_REF_WIDTH) * S
        vid_h = fh * S
        layer = TextLayer()
        for p in active:
            draw_person(frame, layer, p, S, k, vid_h)
        draw_overlay(layer, n_faces, fps, k, vid_h)
        frame = layer.apply(frame)
        pkey = (tuple(summary.items()), tuple(timeline.last.values()),
                int(timeline.elapsed(now) * PANEL_REFRESH_HZ), H, S)
        if pkey != self.panel_key:
            cv = Canvas(PANEL_W, H, S)
            draw_panel(cv, summary, timeline, now)
            self.panel_img, self.panel_key = cv.finish(), pkey
        cv2.imshow("Mascot meter", compose(frame, self.panel_img, S))
        return (cv2.waitKey(1) & 0xFF) != ord("q")


# ---- main ---------------------------------------------------------------------
def parse_size(text: str):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 640x480")


def parse_source(args):
    if args.source is not None:
        s = str(args.source)
        if s == "picam":
            return "picam"
        return int(s) if s.isdigit() else s
    if args.ip:
        return f"http://{args.ip}:81/stream"
    return "picam"


class Meter:
    def __init__(self, args, cloud=None):
        self.args = args
        self.cloud = cloud
        self.store = Store(args.data_dir)
        self.t0 = time.monotonic()
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.raw = RawLog(self.store.raw_dir if args.raw_log else None, self.run_id, self.t0)
        self.pcsv = PersonCSV(self.store.people_dir, self.run_id, self.t0)
        self.tracker = Tracker()
        self.day = None
        self.records = []
        self.timeline = None
        self.summary = None
        self.raw_bytes_cap = int(args.raw_max_gb * 1e9)
        self.min_free = int(args.min_free_gb * 1e9)

    # -- days
    def start_day(self, day: str):
        self.day = day
        self.records = self.store.load_records(day)
        self.timeline = DayTimeline.from_records(day, self.records)
        self.raw.open_day(day)
        self.pcsv.open_day(day)
        self.summary = summarise_records(self.records)
        if self.records:
            log(f"[day] {day}: restored {len(self.records)} people from disk "
                f"(stayed {self.summary['stayed']}, smiles {self.summary['smiles']}, "
                f"laughs {self.summary['laughs']})")
        else:
            log(f"[day] {day}: new day")
        if self.cloud:
            self.cloud.set_day(day, day_start_ms(day), len(self.records))
            self.publish()
        threading.Thread(target=compress_old_raw, args=(self.store.raw_dir, day), daemon=True).start()

    def end_day(self):
        for p in list(self.tracker.people.values()):
            self.record(p)
        self.tracker.people.clear()
        self.store.write_final(self.day, self.records)
        s = summarise_records(self.records)
        log(f"[day] {self.day} finished: {s['n']} people, stayed {s['stayed']}, smiles {s['smiles']}, "
            f"laughs {s['laughs']}, shocked {s['shocked']}")
        if self.cloud:
            self.cloud.day_finished(self.day)

    # -- people
    def record(self, p: Person):
        p.stamp_events(mono_to_ms(p.last_seen))
        p.release()
        self.pcsv.write(p)
        if not p.counted:
            return
        rec = person_record(p)
        rec["q"] = True if self.cloud is None else self.cloud.queue_open()
        self.store.append_record(self.day, rec)
        self.records.append(rec)
        if self.cloud:
            self.cloud.person_added(self.day, len(self.records))
        log(f"[person] P{p.pid}  tracked {p.tracked_seconds:.1f}s  looked {p.eye_time:.1f}s  "
            f"stayed={int(p.stayed)}  react={_f(p.react_time, 1) or 'none'}  "
            f"smiled={int(p.smiled)}  laughed={int(p.laughed)}  shocked={int(p.shock_seen['shocked'])}")

    def compute_summary(self):
        active = [person_record(p) for p in self.tracker.people.values() if p.counted]
        self.summary = summarise_records(self.records + active)
        self.timeline.update(self.summary, now_ms())

    def publish(self):
        if self.cloud:
            self.cloud.set_live(contract_summary(self.summary, self.day), self.timeline.rows)

    # -- main loop
    def run(self):
        args = self.args
        wd = Watchdog()
        source = parse_source(args)
        analyzer = FaceAnalyzer(ensure_model(args.model), args.max_faces)
        if source == "picam":
            grabber = PiCamGrabber(args.size, args.fps, full_fov=not args.crop_fov, rotate=args.rotate)
        else:
            grabber = FrameGrabber(source, args.size, args.fps)
        grabber.start()
        display = Display(args.display_scale) if args.display else None

        self.start_day(day_key(now_ms()))
        self.store.finalise_missing(self.day)
        if self.cloud:
            self.cloud.rescan_days()

        last_t = time.monotonic()
        last_seq = -1
        last_frame_at = time.monotonic()
        fps = 0.0
        detect_ms = 0.0
        last_summary = 0.0
        last_disk = 0.0
        last_stats = time.monotonic()
        last_tune = 0.0
        last_clamp_log = 0.0
        frames_since_stats = 0
        n_faces = 0
        log("[run] running (Ctrl+C to stop)")
        try:
            while True:
                wd.pet()
                mono = time.monotonic()

                # daily rollover at 04:00 SGT
                today = day_key(now_ms())
                if today != self.day:
                    self.end_day()
                    self.start_day(today)

                # disk guard for raw logs
                if self.raw.enabled and mono - last_disk >= DISK_CHECK_EVERY_S:
                    last_disk = mono
                    ok = enforce_disk(self.store.raw_dir, self.raw.path_for(self.day),
                                      self.raw_bytes_cap, self.min_free)
                    if ok == self.raw.paused:
                        self.raw.paused = not ok
                        log("[disk] raw logging " + ("PAUSED: card nearly full" if self.raw.paused else "resumed"))

                # camera health
                since_frame = mono - last_frame_at
                if self.cloud:
                    cam_state = ("starting" if last_seq < 0 else
                                 "no_frames" if since_frame > NO_FRAME_STATUS_S else "ok")
                    self.cloud.set_status(cam_state, fps if cam_state == "ok" else 0.0, n_faces)
                if since_frame > NO_FRAME_EXIT_S:
                    log(f"[camera] no frames for {since_frame:.0f}s, exiting so systemd restarts the camera")
                    sys.exit(3)

                frame, seq = grabber.latest()
                if frame is None or seq == last_seq:
                    if display is not None and cv2.waitKey(5) & 0xFF == ord("q"):
                        break
                    time.sleep(0.005)
                    continue
                last_seq = seq
                last_frame_at = mono
                frame = frame.copy()

                now = time.monotonic()
                raw_dt = now - last_t
                dt = min(raw_dt, MAX_DT)
                if raw_dt > MAX_DT and frames_since_stats and now - last_clamp_log > 60:
                    last_clamp_log = now
                    log(f"[slow] frame gap {raw_dt:.2f}s capped at {MAX_DT}s (durations undercount below ~4 fps)")
                fps = 0.9 * fps + 0.1 * (1.0 / max(raw_dt, 1e-3)) if fps else 1.0 / max(raw_dt, 1e-3)
                last_t = now

                t_det = time.monotonic()
                faces = analyzer.analyze(frame)
                detect_ms = 0.9 * detect_ms + 0.1 * (time.monotonic() - t_det) * 1000
                n_faces = len(faces)
                matched = self.tracker.assign(faces, now)
                ms = now_ms()
                for p in list(self.tracker.people.values()):
                    p.step(matched.get(p.pid), now, dt)
                    p.stamp_events(ms)
                    self.raw.write(seq, now, p)
                lost = self.tracker.retire_lost(now)
                for p in lost:
                    self.record(p)
                    self.tracker.drop(p)

                if lost or now - last_summary >= SUMMARY_EVERY_S:
                    last_summary = now
                    self.compute_summary()
                    self.publish()

                frames_since_stats += 1
                if args.tune and now - last_tune >= 1.0:
                    last_tune = now
                    for p in self.tracker.people.values():
                        if p.face is not None:
                            f = p.face
                            log(f"[tune] P{p.pid} face_height={f.height_frac:.3f} (min {MIN_FACE_HEIGHT_FRAC}) "
                                f"yaw={f.yaw:+.0f} pitch={f.pitch:+.0f} (max {YAW_MAX_DEG:g}/{PITCH_MAX_DEG:g}) "
                                f"engaged={int(p.engaged)} state={p.state}")
                if now - last_stats >= args.stats_every:
                    s = self.summary
                    cloud_txt = self.cloud.describe() if self.cloud else "cloud disabled"
                    log(f"[stats] {frames_since_stats / (now - last_stats):.1f} fps, detect {detect_ms:.0f} ms, "
                        f"faces {n_faces}, tracking {len(self.tracker.people)} | today n={s['n']} "
                        f"stayed={s['stayed']} smiles={s['smiles']} laughs={s['laughs']} "
                        f"shocked={s['shocked']} | {cloud_txt}")
                    last_stats, frames_since_stats = now, 0

                if display is not None:
                    if not display.show(frame, list(self.tracker.people.values()), n_faces, fps,
                                        self.summary, self.timeline, now):
                        break
        finally:
            log("[run] stopping: saving everyone currently tracked")
            for p in list(self.tracker.people.values()):
                try:
                    self.record(p)
                except Exception as e:
                    log(f"[run] could not save P{p.pid}: {e!r}")
            self.tracker.people.clear()
            try:
                self.compute_summary()
                self.publish()
            except Exception:
                pass
            grabber.stop()
            self.raw.close()
            self.pcsv.close()
            analyzer.close()
            if display is not None:
                cv2.destroyAllWindows()


def wait_for_clock(wd: Watchdog, skip: bool):
    if skip:
        log("[clock] NOT waiting for NTP (--no-clock-wait). Timestamps may be wrong.")
        return
    t0 = time.monotonic()
    last_msg = -1e9
    while not clock_synced():
        wd.pet()
        if time.monotonic() - last_msg >= 30:
            last_msg = time.monotonic()
            log(f"[clock] waiting for network time (NTP) before recording... "
                f"({time.monotonic() - t0:.0f}s so far; needs internet)")
        time.sleep(2)
    log(f"[clock] time is synced: {datetime.now(SGT):%Y-%m-%d %H:%M:%S} SGT")


def main():
    global STAY_SECONDS, MIN_FACE_HEIGHT_FRAC, YAW_MAX_DEG, PITCH_MAX_DEG
    ap = argparse.ArgumentParser(description="Mascot engagement + happiness meter (Raspberry Pi)")
    ap.add_argument("--source", help="'picam' (default), a stream URL, or a webcam index such as 0")
    ap.add_argument("--ip", help="ESP32 IP address; stream is http://<ip>:81/stream")
    ap.add_argument("--size", type=parse_size, default=(640, 480), help="camera frame size (default 640x480)")
    ap.add_argument("--fps", type=float, default=12.0, help="camera frame rate (default 12)")
    ap.add_argument("--rotate", type=int, choices=(0, 180), default=0, help="180 if the camera is mounted upside down")
    ap.add_argument("--crop-fov", action="store_true",
                    help="use the camera's cropped (narrower) sensor mode instead of the full field of view")
    ap.add_argument("--max-faces", type=int, default=5, help="max faces tracked at once (default 5)")
    ap.add_argument("--stay-seconds", type=float, default=STAY_SECONDS)
    ap.add_argument("--min-face-frac", type=float, default=MIN_FACE_HEIGHT_FRAC,
                    help=f"min face height / frame height to count as looking (default {MIN_FACE_HEIGHT_FRAC})")
    ap.add_argument("--yaw-max", type=float, default=YAW_MAX_DEG, help="max head turn left/right in degrees")
    ap.add_argument("--pitch-max", type=float, default=PITCH_MAX_DEG, help="max head tilt up/down in degrees")
    ap.add_argument("--data-dir", default=os.path.join(APP_DIR, "data"))
    ap.add_argument("--no-raw-log", dest="raw_log", action="store_false", help="disable the raw blendshape CSV")
    ap.add_argument("--raw-max-gb", type=float, default=6.0, help="total size cap for raw logs (default 6 GB)")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="always keep this much of the SD card free")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--no-cloud", action="store_true", help="do not upload (local only)")
    ap.add_argument("--no-clock-wait", action="store_true", help="testing only: do not wait for NTP")
    ap.add_argument("--tune", action="store_true", help="print face size and head angles every second")
    ap.add_argument("--stats-every", type=float, default=60.0, help="seconds between [stats] log lines")
    ap.add_argument("--display", action="store_true", help="show the local window (needs a desktop)")
    ap.add_argument("--display-scale", type=float, default=1.0)
    args = ap.parse_args()
    STAY_SECONDS = args.stay_seconds
    MIN_FACE_HEIGHT_FRAC = args.min_face_frac
    YAW_MAX_DEG, PITCH_MAX_DEG = args.yaw_max, args.pitch_max

    # systemd stop sends SIGTERM: turn it into a normal exit so everyone tracked gets saved
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sd_notify("READY=1")
    wd = Watchdog()
    log(f"[start] size {args.size[0]}x{args.size[1]}, fps {args.fps:g}, min face {MIN_FACE_HEIGHT_FRAC}, "
        f"yaw/pitch max {YAW_MAX_DEG:g}/{PITCH_MAX_DEG:g}, stay {STAY_SECONDS:g}s, data {args.data_dir}")
    wait_for_clock(wd, args.no_clock_wait)

    cloud = None
    if not args.no_cloud:
        import mascot_cloud
        cloud = mascot_cloud.Cloud.from_env(args.data_dir)
        if cloud is None:
            log("[cloud] no Firebase settings found (/etc/mascot/firebase.env); running local only")
        else:
            cloud.start()
    meter = Meter(args, cloud)
    try:
        meter.run()
    except KeyboardInterrupt:
        pass
    finally:
        if cloud:
            cloud.stop()
        sd_notify("STOPPING=1")


if __name__ == "__main__":
    main()
