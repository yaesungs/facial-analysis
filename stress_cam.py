"""Webcam stress detector for games (stress-test-gaming-team).

  python stress_cam.py                  detect stress live (calibrates first)
  python stress_cam.py record calm      record a labeled session for training
  python stress_cam.py record stressed
  python stress_cam.py train            train your personal model on the recordings

While running, the stress level is sent as JSON over UDP (127.0.0.1:5005 by
default) so a game in any engine can react to it; see game_receiver_example.py.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from face_features import FaceDetector, FeatureExtractor, ensure_model
from pulse import PulseEstimator
from stress_model import LABELS, READABLE, BaselineScorer, PersonalModel, Recorder, Smoother, level, train

HERE = Path(__file__).resolve().parent
FACE_MODEL = HERE / "models" / "face_landmarker.task"
PERSONAL_MODEL = HERE / "models" / "personal_model.joblib"
RECORDINGS = HERE / "recordings"

WINDOW = "Stress detector"
RECORD_WARMUP = 10.0  # seconds before a recording starts, so blink rate and pulse have data
HISTORY_SECONDS = 60

# OpenCV colors are (blue, green, red).
GREEN, YELLOW, RED = (90, 200, 90), (0, 200, 255), (60, 60, 235)
WHITE, GRAY = (255, 255, 255), (160, 160, 160)
LEVEL_COLORS = {"calm": GREEN, "tense": YELLOW, "stressed": RED}


class StressApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.detector = FaceDetector(ensure_model(FACE_MODEL))
        self.pulse = None if args.no_pulse else PulseEstimator()
        self.extractor = FeatureExtractor(pulse=self.pulse)
        self.recorder = self.scorer = self.udp = None
        if args.command == "record":
            self.recorder = Recorder(RECORDINGS, args.label)
            print(f"Recording '{args.label}' for {args.seconds:.0f} s after a {RECORD_WARMUP:.0f} s warm-up.")
        else:
            if PERSONAL_MODEL.exists() and not args.baseline:
                self.scorer = PersonalModel(PERSONAL_MODEL)
                print("Using your personal model (add --baseline to calibrate instead).")
            else:
                self.scorer = BaselineScorer(args.calibration, args.sensitivity)
                print(f"Calibrating for {args.calibration:.0f} s: relax and look at the screen like you normally do.")
            self.smoother = Smoother(args.smoothing)
            if args.udp != "off":
                self.udp = UdpSender(args.udp)
                print(f"Sending stress values to udp://{args.udp}")

        self.t, self.face, self.feats = 0.0, None, None
        self.stress, self.reasons = None, {}
        self.history = deque()
        self.fps, self.details = 0.0, False

    def run(self) -> None:
        if not self.args.no_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        last_frame, last_print = None, -math.inf
        try:
            for t, frame in frames(self.args):
                self.step(t, frame)
                now = time.perf_counter()
                if last_frame is not None:
                    rate = 1.0 / max(now - last_frame, 1e-6)
                    self.fps = 0.9 * self.fps + 0.1 * rate if self.fps else rate
                last_frame = now
                if self.args.no_window:
                    if t - last_print >= 1.0:
                        last_print = t
                        print(self.status())
                elif not self.show(frame):
                    break
                if self.done():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.detector.close()
            cv2.destroyAllWindows()
            if self.recorder:
                self.recorder.close()
                if self.recorder.rows:
                    print(f"Saved {self.recorder.rows} samples to {self.recorder.path}")
                else:
                    print("Nothing was recorded (no face seen after the warm-up).")

    def step(self, t: float, frame: np.ndarray) -> None:
        self.t = t
        self.face = self.detector.detect(frame, t)
        self.extractor.update(t, frame, self.face)
        self.feats = self.extractor.features(t)

        if self.recorder:
            if self.feats and t >= RECORD_WARMUP:
                self.recorder.write(t, self.feats)
            return
        if self.feats:
            if self.scorer.ready:
                value, self.reasons = self.scorer.score(self.feats)
                self.stress = self.smoother.update(t, value)
            else:
                self.scorer.calibrate(t, self.feats)
        if self.stress is not None:
            self.history.append((t, self.stress))
            while t - self.history[0][0] > HISTORY_SECONDS:
                self.history.popleft()
        if self.udp:
            self.udp.send(t, self.message())

    def done(self) -> bool:
        if self.recorder and self.t >= RECORD_WARMUP + self.args.seconds:
            return True
        return self.args.max_seconds is not None and self.t >= self.args.max_seconds

    def message(self) -> dict:
        """What the game receives, up to 20 times per second."""
        def number(name):
            value = self.feats[name] if self.feats else math.nan
            return None if math.isnan(value) else round(value, 1)

        return {
            "stress": None if self.stress is None else round(self.stress, 3),
            "level": None if self.stress is None else level(self.stress),
            "face": self.face is not None,
            "calibrating": not self.scorer.ready,
            "heart_rate": number("heart_rate"),
            "blink_rate": number("blink_rate"),
            "time": round(self.t, 2),
        }

    def top_reasons(self, n: int = 2) -> list[str]:
        ranked = sorted(self.reasons.items(), key=lambda kv: -kv[1])
        return [READABLE[name] for name, push in ranked[:n] if push > 0.25]

    def status(self) -> str:
        """One console line, printed every second with --no-window."""
        head = f"[{self.t:6.1f}s]" + ("" if self.face else " (no face)")
        if self.recorder:
            if self.t < RECORD_WARMUP:
                return f"{head} recording starts in {RECORD_WARMUP - self.t:.0f} s"
            return f"{head} recording '{self.recorder.label}': {self.recorder.rows} samples"
        if not self.scorer.ready:
            return f"{head} calibrating {self.scorer.progress(self.t):.0%}"
        if self.stress is None:
            return f"{head} waiting for a face"
        m = {k: "--" if v is None else v for k, v in self.message().items()}
        return (f"{head} stress {self.stress:.2f} {m['level']:<8} heart {m['heart_rate']} bpm, "
                f"blinks {m['blink_rate']}/min, why: {', '.join(self.top_reasons()) or '-'}")

    # ---- window ---------------------------------------------------------------

    def show(self, frame: np.ndarray) -> bool:
        """Draws the overlay and handles keys. Returns False when the user quits."""
        self.draw(frame)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key == ord("d"):
            self.details = not self.details
        if key == ord("c") and isinstance(self.scorer, BaselineScorer):
            self.scorer.restart()
            self.smoother.reset()
            self.stress, self.reasons = None, {}
            self.history.clear()
        return True

    def draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        color = LEVEL_COLORS[level(self.stress)] if self.stress is not None else GRAY
        if self.face is not None:
            x0, y0, x1, y1 = self.face.box()
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            if self.details and self.pulse is not None:
                for bx0, by0, bx1, by1 in self.pulse.rois:
                    cv2.rectangle(img, (bx0, by0), (bx1, by1), (255, 0, 255), 1)
        else:
            put(img, "No face - look at the camera", (w // 2 - 160, h // 2), 0.7, YELLOW, 2)

        shade(img, 0, 0, 270, 136)
        if self.recorder:
            self._draw_recording(img)
        elif not self.scorer.ready:
            put(img, "Calibrating...", (12, 32), 0.8, YELLOW, 2)
            put(img, "relax, look at the screen", (12, 58))
            bar(img, 12, 70, 246, self.scorer.progress(self.t), YELLOW)
        elif self.stress is not None:
            put(img, f"STRESS {self.stress:4.0%}", (12, 32), 0.8, color, 2)
            bar(img, 12, 44, 246, self.stress, color)
            put(img, level(self.stress).upper(), (12, 82), 0.6, color, 2)
            put(img, ", ".join(self.top_reasons()), (12, 104), 0.45)

        hr = self.feats["heart_rate"] if self.feats else math.nan
        blinks = self.feats["blink_rate"] if self.feats else math.nan
        put(img, f"heart {'--' if math.isnan(hr) else f'{hr:.0f}'} bpm (beta)   "
                 f"blinks {'--' if math.isnan(blinks) else f'{blinks:.0f}'}/min", (12, 126), 0.42, GRAY)

        mode = "recording" if self.recorder else self.scorer.name
        put_right(img, f"{mode} | {self.fps:.0f} fps", 20)
        keys = "q quit  d details" + ("  c recalibrate" if isinstance(self.scorer, BaselineScorer) else "")
        put_right(img, keys, 40)
        if self.details:
            self._draw_details(img)
        if not self.recorder:
            self._draw_history(img)

    def _draw_recording(self, img: np.ndarray) -> None:
        label = self.recorder.label
        if self.t < RECORD_WARMUP:
            put(img, f"Get ready: {RECORD_WARMUP - self.t:.0f}", (12, 32), 0.8, YELLOW, 2)
            put(img, f"then recording '{label}'", (12, 58))
        else:
            elapsed = self.t - RECORD_WARMUP
            put(img, f"REC {label}", (12, 32), 0.8, RED, 2)
            bar(img, 12, 44, 246, elapsed / self.args.seconds, RED)
            put(img, f"{elapsed:.0f} / {self.args.seconds:.0f} s   {self.recorder.rows} samples", (12, 82))

    def _draw_details(self, img: np.ndarray) -> None:
        if not self.feats:
            return
        w = img.shape[1]
        x0, y0 = w - 200, 52
        shade(img, x0, y0, w, y0 + 18 * len(self.feats) + 8)
        for i, (name, value) in enumerate(self.feats.items()):
            y = y0 + 18 * (i + 1)
            put(img, name, (x0 + 8, y), 0.42, GRAY)
            put(img, "--" if math.isnan(value) else f"{value:.2f}", (x0 + 140, y), 0.42)

    def _draw_history(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = 10, h - 60, w - 10, h - 10
        shade(img, x0, y0, x1, y1)
        for threshold in (0.35, 0.65):
            y = int(y1 - threshold * (y1 - y0))
            cv2.line(img, (x0, y), (x1, y), (80, 80, 80), 1)
        points = [(int(x1 - (self.t - t) / HISTORY_SECONDS * (x1 - x0)), int(y1 - s * (y1 - y0)))
                  for t, s in self.history]
        if len(points) > 1:
            cv2.polylines(img, [np.array(points, np.int32)], False, WHITE, 1, cv2.LINE_AA)
        put(img, f"stress, last {HISTORY_SECONDS} s", (x0 + 6, y0 + 14), 0.4, GRAY)


class UdpSender:
    """Sends JSON messages to a game over UDP, at most `rate` times per second."""

    def __init__(self, address: str, rate: float = 20.0):
        host, port = address.rsplit(":", 1)
        self.target = (host, int(port))
        self.interval, self._last = 1.0 / rate, -math.inf
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, t: float, message: dict) -> None:
        if t - self._last < self.interval:
            return
        self._last = t
        try:
            self._sock.sendto(json.dumps(message).encode(), self.target)
        except OSError:
            pass  # nobody is listening yet; that's fine


def frames(args: argparse.Namespace):
    """Yields (seconds since start, BGR image) from the webcam or a video file."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video file {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = open_camera(args.camera)
    start, index, failures = time.monotonic(), 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    return
                failures += 1
                if failures > 50:
                    raise SystemExit("The webcam stopped sending frames.")
                time.sleep(0.02)
                continue
            failures = 0
            if args.video:
                yield index / fps, frame  # video time, so results don't depend on processing speed
                index += 1
            else:
                yield time.monotonic() - start, cv2.flip(frame, 1)  # mirrored, like a selfie
    finally:
        cap.release()


def open_camera(index: int) -> cv2.VideoCapture:
    # DirectShow opens much faster than the default backend on many Windows laptops.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    raise SystemExit(f"Could not open webcam {index}. Try --camera 1, and close other apps using the camera.")


def put(img, text, org, scale=0.5, color=WHITE, thickness=1) -> None:
    """Text with a dark drop shadow so it is readable on any background."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_right(img, text, y, scale=0.45, color=GRAY) -> None:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    put(img, text, (img.shape[1] - width - 10, y), scale, color)


def shade(img, x0, y0, x1, y1, darkness=0.55) -> None:
    region = img[y0:y1, x0:x1]
    region[:] = (region * (1 - darkness)).astype(np.uint8)


def bar(img, x, y, width, fraction, color, height=12) -> None:
    fraction = min(max(fraction, 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + width, y + height), GRAY, 1)
    cv2.rectangle(img, (x, y), (x + int(width * fraction), y + height), color, -1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera", type=int, default=0, help="webcam number (default 0)")
    common.add_argument("--video", help="read a video file instead of the webcam")
    common.add_argument("--no-window", action="store_true", help="no camera window; print to the console")
    common.add_argument("--no-pulse", action="store_true", help="turn off the experimental heart-rate estimate")
    common.add_argument("--max-seconds", type=float, help="stop after this many seconds")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    live = commands.add_parser("run", parents=[common], help="detect stress live (the default)")
    live.add_argument("--baseline", action="store_true", help="ignore your personal model and calibrate instead")
    live.add_argument("--calibration", type=float, default=20.0, help="calibration length in seconds (default 20)")
    live.add_argument("--sensitivity", type=float, default=0.4, help="how strongly the baseline score reacts (default 0.4)")
    live.add_argument("--smoothing", type=float, default=0.0,
                      help="extra smoothing of the output in seconds (default 0 = react instantly)")
    live.add_argument("--udp", default="127.0.0.1:5005", help="host:port to send stress values to, or 'off'")
    record = commands.add_parser("record", parents=[common], help="record a labeled session for training")
    record.add_argument("label", choices=LABELS)
    record.add_argument("--seconds", type=float, default=120.0, help="length of the recording (default 120)")
    commands.add_parser("train", help="train your personal model on everything in recordings/")

    if not argv or argv[0] not in ("run", "record", "train", "-h", "--help"):
        argv = ["run", *argv]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.command == "train":
        train(RECORDINGS, PERSONAL_MODEL)
    else:
        StressApp(args).run()


if __name__ == "__main__":
    main()
