"""Fake stress detector, so the game can be built and tested without a webcam.

  python stress_simulator.py              window with a slider: drag it to set the stress
  python stress_simulator.py --wave       stress rises and falls by itself
  python stress_simulator.py --no-window  console only (always waves)

Sends exactly the messages stress_cam.py sends (see API.md) to 127.0.0.1:5005, with
"source": "simulator". Keys in the window: c = calibrating on/off, f = face lost on/off,
w = wave on/off, q = quit.
"""

import argparse
import math
import time

import cv2
import numpy as np

from stress_api import FACE_PORT, StressSender, level

WINDOW = "Stress simulator"
SLIDER = "stress %"
LEVEL_COLORS = {"calm": (90, 200, 90), "tense": (0, 200, 255), "stressed": (60, 60, 235)}  # BGR


def draw(stress: float, face: bool, calibrating: bool, wave: bool) -> np.ndarray:
    img = np.full((170, 460, 3), 28, np.uint8)
    if calibrating:
        title, color = "CALIBRATING (stress = null)", (0, 200, 255)
    else:
        title, color = f"STRESS {stress:4.0%}  {level(stress).upper()}", LEVEL_COLORS[level(stress)]
    cv2.putText(img, title, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (14, 50), (446, 66), (70, 70, 70), -1)
    cv2.rectangle(img, (14, 50), (14 + int(432 * stress), 66), color, -1)
    status = f"face {'visible' if face else 'LOST'}   wave {'on' if wave else 'off'}   -> udp 127.0.0.1:{FACE_PORT}"
    cv2.putText(img, status, (14, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "drag the slider | c calibrating  f face  w wave  q quit", (14, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wave", action="store_true", help="stress goes up and down by itself")
    parser.add_argument("--period", type=float, default=20.0, help="seconds per wave (default 20)")
    parser.add_argument("--no-window", action="store_true", help="no window; print to the console (waves)")
    parser.add_argument("--udp", default=f"127.0.0.1:{FACE_PORT}", help="host:port to send to")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    args = parser.parse_args()

    sender = StressSender(args.udp)
    stress, face, calibrating, wave = 0.0, True, False, args.wave or args.no_window
    if not args.no_window:
        cv2.namedWindow(WINDOW)
        cv2.createTrackbar(SLIDER, WINDOW, 0, 100, lambda value: None)
    start, last_print = time.monotonic(), -math.inf
    while True:
        t = time.monotonic() - start
        if wave:
            stress = 0.5 - 0.5 * math.cos(2 * math.pi * t / args.period)
            if not args.no_window:
                cv2.setTrackbarPos(SLIDER, WINDOW, round(100 * stress))
        elif not args.no_window:
            stress = cv2.getTrackbarPos(SLIDER, WINDOW) / 100
        sender.send(t, {
            "source": "simulator", "time": round(t, 2),
            "stress": None if calibrating else round(stress, 3),
            "level": None if calibrating else level(stress),
            "face": face, "calibrating": calibrating, "heart_rate": None, "blink_rate": None,
        })
        if args.no_window:
            time.sleep(0.02)
            if t - last_print >= 1.0:
                last_print = t
                print(f"[{t:6.1f}s] sending stress {stress:.2f} {level(stress)}")
        else:
            cv2.imshow(WINDOW, draw(stress, face, calibrating, wave))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            calibrating ^= key == ord("c")
            face ^= key == ord("f")
            wave ^= key == ord("w")
        if args.seconds and t >= args.seconds:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
