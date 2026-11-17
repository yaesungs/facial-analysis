"""Live Muse 2 EEG viewer (stress-test-gaming-team), separate from the webcam detector.

  python muse_eeg.py                    connect to a Muse 2 over Bluetooth and show live EEG
  python muse_eeg.py --name Muse-1A2B   connect to one specific headband
  python muse_eeg.py --synthetic        no headband: BrainFlow's simulated board, for testing

The window shows the four electrodes' signals, how good each one's skin contact is,
how the signal power splits into the classic brainwave bands, and an EEG stress index
compared with a short relaxed calibration. The same values go out as JSON over UDP
(127.0.0.1:5006), next to the face detector's 5005, so a game can use both.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from collections import deque

import cv2
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

WINDOW = "Muse 2 EEG"
NAMES = ("TP9", "AF7", "AF8", "TP10")
SITES = {"TP9": "left ear", "AF7": "left forehead", "AF8": "right forehead", "TP10": "right ear"}
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 44)}
BAND_MEANING = {"delta": "deep sleep, blinks", "theta": "drowsy, daydream", "alpha": "relaxed",
                "beta": "focus, tension", "gamma": "muscle, intense"}
BUFFER_SECONDS = 10  # filtered as a whole, so filter edge effects stay off screen
SHOW_SECONDS = 5     # visible part of the traces
POWER_SECONDS = 2    # band powers and the stress index use the last 2 s
HISTORY_SECONDS = 60

# OpenCV colors are (blue, green, red).
GREEN, YELLOW, RED = (90, 200, 90), (0, 200, 255), (60, 60, 235)
WHITE, GRAY, PANEL, BACKGROUND = (235, 235, 235), (150, 150, 150), (40, 40, 40), (24, 24, 24)
TRACE_COLORS = ((255, 170, 80), (120, 220, 120), (120, 200, 255), (220, 140, 255))
BAND_COLORS = {"delta": (200, 130, 90), "theta": (200, 190, 90), "alpha": (110, 210, 110),
               "beta": (60, 170, 240), "gamma": (110, 110, 240)}
QUALITY_COLORS = {"good": GREEN, "fair": YELLOW, "poor": RED, "flat": RED}
LEVEL_COLORS = {"calm": GREEN, "tense": YELLOW, "stressed": RED}

CONNECT_TIPS = """Check that:
  - the Muse is charged and on (hold its button until the lights scroll)
  - it is not connected to the Muse app on your phone (close the app or turn off phone Bluetooth)
  - Bluetooth is on in Windows (no need to pair the Muse in Windows settings)
  - if there are several Muses around, pass --name Muse-XXXX (the name the Muse app shows)"""


def contact(sd: float) -> str:
    """Contact quality from the spread of the cleaned signal (uV). Brain signals on a Muse
    are roughly 5-30 uV; a loose sensor, hair or movement gives hundreds."""
    if sd < 0.3:
        return "flat"
    return "good" if sd <= 40 else "fair" if sd <= 100 else "poor"


def level(stress: float) -> str:
    return "calm" if stress < 0.35 else "tense" if stress < 0.65 else "stressed"


class EegAnalyzer:
    """Keeps the last seconds of EEG; turns them into contact quality, band powers and an index."""

    def __init__(self, rate: int, mains_hz: float):
        self.rate = rate
        self._buffer = np.zeros((len(NAMES), 0))
        self._bandpass = butter(4, [1, 45], btype="band", fs=rate, output="sos")
        self._notch = iirnotch(mains_hz, 30, fs=rate)  # power-line hum

    def add(self, samples: np.ndarray) -> None:
        self._buffer = np.concatenate([self._buffer, samples], axis=1)[:, -BUFFER_SECONDS * self.rate:]

    def analyze(self) -> dict | None:
        if self._buffer.shape[1] < POWER_SECONDS * self.rate:
            return None
        clean = filtfilt(*self._notch, sosfiltfilt(self._bandpass, self._buffer, axis=1), axis=1)
        recent = clean[:, -POWER_SECONDS * self.rate:]
        sd = recent.std(axis=1)
        quality = [contact(s) for s in sd]
        usable = [i for i, q in enumerate(quality) if q in ("good", "fair")]
        freqs, psd = welch(recent, fs=self.rate, nperseg=self.rate, axis=1)  # 1 Hz resolution
        result = {"traces": clean[:, -SHOW_SECONDS * self.rate:], "sd": sd, "quality": quality,
                  "usable": usable, "bands": None, "index": math.nan}
        if usable:
            power = {band: float(psd[usable][:, (freqs >= lo) & (freqs < hi)].sum(axis=1).mean())
                     for band, (lo, hi) in BANDS.items()}
            total = sum(power.values())
            result["bands"] = {band: p / total for band, p in power.items()}
            result["index"] = power["beta"] / (power["alpha"] + power["theta"])
        return result


class EegStress:
    """EEG stress index compared with your relaxed baseline, mapped to 0..1 like the face score.

    The index is beta / (alpha + theta) power: it rises with mental effort and arousal and
    falls when you relax (alpha grows). Face and jaw muscle tension also adds beta, so
    frowning and clenching push it up as well.
    """

    def __init__(self, calibration_seconds: float = 20.0, sensitivity: float = 0.4):
        self.calibration_seconds = calibration_seconds
        self.sensitivity = sensitivity
        self.restart()

    def restart(self) -> None:
        self.mean = self.std = None
        self._samples, self._started = [], None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def progress(self, t: float) -> float:
        return 0.0 if self._started is None else min((t - self._started) / self.calibration_seconds, 1.0)

    def update(self, t: float, index: float) -> float | None:
        """Feeds one index value. Returns stress 0..1 once calibrated, None while calibrating."""
        value = math.log(index)
        if not self.ready:
            if self._started is None:
                self._started = t
            self._samples.append(value)
            if t - self._started >= self.calibration_seconds and len(self._samples) >= 20:
                self.mean = float(np.mean(self._samples))
                self.std = max(float(np.std(self._samples)), 0.1)  # changes under ~10% never count
            return None
        z = (value - self.mean) / self.std
        return 1.0 - math.exp(-self.sensitivity * min(max(z - 1.0, 0.0), 6.0))  # first "wiggle" ignored


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


class MuseViewer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        if args.synthetic:
            board_id, self.source = BoardIds.SYNTHETIC_BOARD, "simulated headset"
        else:
            board_id, self.source = BoardIds.MUSE_2_BOARD, args.name or "Muse 2"
            params.serial_number = args.name or ""
            params.timeout = args.timeout
        self.board = BoardShim(board_id, params)
        self.rate = BoardShim.get_sampling_rate(board_id)
        self.channels = BoardShim.get_eeg_channels(board_id)[:len(NAMES)]
        self.analyzer = EegAnalyzer(self.rate, args.mains)
        self.scorer = EegStress(args.calibration, args.sensitivity)
        self.udp = None if args.udp == "off" else UdpSender(args.udp)
        self.t, self.last_data = 0.0, 0.0
        self.result, self.stress = None, None
        self.history = deque()

    def run(self) -> None:
        print(f"Connecting to {self.source} ...")
        try:
            self.board.prepare_session()
            self.board.start_stream()
        except BrainFlowError as error:
            raise SystemExit(f"Could not connect to the Muse ({error}).\n{CONNECT_TIPS}")
        print(f"Connected. Calibrating for {self.args.calibration:.0f} s: sit relaxed with your eyes open.")
        if not self.args.no_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        start, last_print = time.monotonic(), -math.inf
        try:
            while True:
                self.t = time.monotonic() - start
                self.step()
                if self.args.no_window:
                    time.sleep(0.03)
                    if self.t - last_print >= 1.0:
                        last_print = self.t
                        print(self.status())
                elif not self.show():
                    break
                if self.args.max_seconds and self.t >= self.args.max_seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.board.stop_stream()
            except BrainFlowError:
                pass
            self.board.release_session()
            cv2.destroyAllWindows()

    def step(self) -> None:
        data = self.board.get_board_data()  # every sample that arrived since the last call
        if not data.shape[1]:
            return
        self.last_data = self.t
        self.analyzer.add(data[self.channels])
        self.result = self.analyzer.analyze()
        self.stress = None
        if self.result and len(self.result["usable"]) >= 2 and self.result["index"] > 0:
            self.stress = self.scorer.update(self.t, self.result["index"])
        if self.stress is not None:
            self.history.append((self.t, self.stress))
        while self.history and self.t - self.history[0][0] > HISTORY_SECONDS:
            self.history.popleft()
        if self.udp:
            self.udp.send(self.t, self.message())

    def message(self) -> dict:
        """What the game receives on port 5006."""
        r = self.result
        return {
            "source": "eeg",
            "stress": None if self.stress is None else round(self.stress, 3),
            "level": None if self.stress is None else level(self.stress),
            "calibrating": not self.scorer.ready,
            "contact": dict(zip(NAMES, r["quality"])) if r else None,
            "bands": {band: round(v, 3) for band, v in r["bands"].items()} if r and r["bands"] else None,
            "time": round(self.t, 2),
        }

    def problem(self) -> str | None:
        """What stops a measurement right now, in words, or None."""
        if self.t - self.last_data > 3:
            return "No data from the headband - is it on and in range?"
        if self.result is None:
            return "Collecting the first seconds of data..."
        bad = [name for name, q in zip(NAMES, self.result["quality"]) if q in ("poor", "flat")]
        if bad:
            return f"Fix contact: {', '.join(bad)} - move hair aside, press the band closer, give it ~10 s to settle"
        return None

    def status(self) -> str:
        """One console line, printed every second with --no-window."""
        head = f"[{self.t:6.1f}s]"
        if self.result is None:
            return f"{head} {self.problem()}"
        contacts = " ".join(f"{n}:{q}" for n, q in zip(NAMES, self.result["quality"]))
        bands = self.result["bands"]
        waves = f"alpha {bands['alpha']:.0%} beta {bands['beta']:.0%}" if bands else "bands --"
        if not self.scorer.ready:
            score = f"calibrating {self.scorer.progress(self.t):.0%}"
        elif self.stress is None:
            score = "stress -- (need 2+ good sensors)"
        else:
            score = f"stress {self.stress:.2f} {level(self.stress)}"
        return f"{head} {contacts} | {waves} | {score}"

    # ---- window ---------------------------------------------------------------

    def show(self) -> bool:
        cv2.imshow(WINDOW, self.draw())
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key == ord("c"):
            self.scorer.restart()
            self.stress = None
            self.history.clear()
        return True

    def draw(self) -> np.ndarray:
        img = np.full((620, 980, 3), BACKGROUND, np.uint8)
        put(img, "Muse 2 EEG", (16, 32), 0.8, WHITE, 2)
        put(img, f"{self.source}  |  {self.rate} Hz", (186, 32), 0.5, GRAY)
        put_right(img, "q quit  c recalibrate", 32)
        self._draw_traces(img, 16, 52, 640)
        self._draw_bands(img, 670, 52)
        self._draw_stress(img, 670, 330)
        problem = self.problem()
        if problem:
            put(img, problem, (16, 606), 0.5, YELLOW)
        return img

    def _draw_traces(self, img, x0, y0, x1, lane=124, gap=10) -> None:
        for i, name in enumerate(NAMES):
            top = y0 + i * (lane + gap)
            mid = top + lane // 2
            cv2.rectangle(img, (x0, top), (x1, top + lane), PANEL, -1)
            cv2.line(img, (x0, mid), (x1, mid), (60, 60, 60), 1)
            put(img, f"{name}  {SITES[name]}", (x0 + 10, top + 20), 0.5, TRACE_COLORS[i])
            if self.result is None:
                continue
            quality, sd = self.result["quality"][i], self.result["sd"][i]
            put_right(img, f"{quality}  {sd:.0f} uV", top + 20, 0.45, QUALITY_COLORS[quality], right=x1 - 10)
            trace = self.result["traces"][i]
            ys = mid - np.clip(trace / 100.0, -1, 1) * (lane / 2 - 6)  # +-100 uV fills the lane
            xs = np.linspace(x0, x1, len(trace))
            cv2.polylines(img, [np.stack([xs, ys], axis=1).astype(np.int32)], False,
                          TRACE_COLORS[i], 1, cv2.LINE_AA)
        put(img, f"last {SHOW_SECONDS} s, +-100 uV per lane", (x0, y0 + 4 * (lane + gap) + 6), 0.4, GRAY)

    def _draw_bands(self, img, x0, y0, width=294) -> None:
        cv2.rectangle(img, (x0, y0), (x0 + width, y0 + 262), PANEL, -1)
        put(img, "Brainwave bands (% of power)", (x0 + 12, y0 + 24), 0.5, WHITE)
        bands = self.result["bands"] if self.result else None
        for i, band in enumerate(BANDS):
            y = y0 + 52 + i * 40
            put(img, band, (x0 + 12, y + 4), 0.5, BAND_COLORS[band])
            put(img, BAND_MEANING[band], (x0 + 12, y + 20), 0.38, GRAY)
            cv2.rectangle(img, (x0 + 140, y - 10), (x0 + 240, y + 6), (60, 60, 60), -1)
            if bands:
                fill = int(100 * min(bands[band] / 0.6, 1.0))  # 60% of the power fills the bar
                cv2.rectangle(img, (x0 + 140, y - 10), (x0 + 140 + fill, y + 6), BAND_COLORS[band], -1)
                put(img, f"{bands[band]:.0%}", (x0 + 248, y + 4), 0.45, WHITE)
        put(img, "eyes closed -> alpha should rise", (x0 + 12, y0 + 252), 0.4, GRAY)

    def _draw_stress(self, img, x0, y0, width=294) -> None:
        cv2.rectangle(img, (x0, y0), (x0 + width, y0 + 252), PANEL, -1)
        if not self.scorer.ready:
            put(img, "Calibrating...", (x0 + 12, y0 + 34), 0.7, YELLOW, 2)
            put(img, "sit relaxed, eyes open", (x0 + 12, y0 + 60), 0.45, GRAY)
            bar(img, x0 + 12, y0 + 74, width - 24, self.scorer.progress(self.t), YELLOW)
        elif self.stress is None:
            put(img, "EEG STRESS  --", (x0 + 12, y0 + 34), 0.7, GRAY, 2)
            put(img, "needs 2+ sensors with good contact", (x0 + 12, y0 + 60), 0.42, GRAY)
        else:
            color = LEVEL_COLORS[level(self.stress)]
            put(img, f"EEG STRESS {self.stress:4.0%}", (x0 + 12, y0 + 34), 0.7, color, 2)
            bar(img, x0 + 12, y0 + 46, width - 24, self.stress, color)
            put(img, level(self.stress).upper(), (x0 + 12, y0 + 88), 0.55, color, 2)
        # stress history
        gx0, gy0, gx1, gy1 = x0 + 12, y0 + 120, x0 + width - 12, y0 + 236
        cv2.rectangle(img, (gx0, gy0), (gx1, gy1), (30, 30, 30), -1)
        for threshold in (0.35, 0.65):
            y = int(gy1 - threshold * (gy1 - gy0))
            cv2.line(img, (gx0, y), (gx1, y), (70, 70, 70), 1)
        points = [(int(gx1 - (self.t - t) / HISTORY_SECONDS * (gx1 - gx0)), int(gy1 - s * (gy1 - gy0)))
                  for t, s in self.history]
        if len(points) > 1:
            cv2.polylines(img, [np.array(points, np.int32)], False, WHITE, 1, cv2.LINE_AA)
        put(img, f"EEG stress, last {HISTORY_SECONDS} s", (gx0 + 6, gy0 + 16), 0.4, GRAY)


def put(img, text, org, scale=0.5, color=WHITE, thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_right(img, text, y, scale=0.45, color=GRAY, right=None) -> None:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    put(img, text, ((img.shape[1] - 16 if right is None else right) - width, y), scale, color)


def bar(img, x, y, width, fraction, color, height=14) -> None:
    fraction = min(max(fraction, 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + width, y + height), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(width * fraction), y + height), color, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="connect to this Muse, e.g. Muse-1A2B (default: the first one found)")
    parser.add_argument("--synthetic", action="store_true", help="use a simulated headset instead of a Muse")
    parser.add_argument("--timeout", type=int, default=15, help="seconds to search for the Muse (default 15)")
    parser.add_argument("--mains", type=float, default=60.0,
                        help="power-line frequency to filter out: 60 (US, Korea) or 50 (Europe)")
    parser.add_argument("--calibration", type=float, default=20.0, help="calibration length in seconds (default 20)")
    parser.add_argument("--sensitivity", type=float, default=0.4, help="how strongly the stress score reacts (default 0.4)")
    parser.add_argument("--udp", default="127.0.0.1:5006", help="host:port to send EEG values to, or 'off'")
    parser.add_argument("--no-window", action="store_true", help="no window; print to the console")
    parser.add_argument("--max-seconds", type=float, help="stop after this many seconds")
    return parser.parse_args()


if __name__ == "__main__":
    MuseViewer(parse_args()).run()
