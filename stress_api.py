"""The stress API: how the detector talks to games. The message format is in API.md.

Detector side (stress_cam.py, stress_simulator.py):

    sender = StressSender()                    # sends to 127.0.0.1:5005
    sender.send(t, {"source": "face", "stress": 0.63, "level": "tense", ...})

Game side (pygame, or any Python game):

    from stress_api import StressClient
    stress = StressClient()                    # listens on 127.0.0.1:5005
    while running:                             # your game loop
        stress.update()                        # never blocks
        jump = 1.0 + 0.5 * stress.value        # 0.0 calm .. 1.0 stressed
"""

from __future__ import annotations

import json
import math
import socket
import time

VERSION = 1
FACE_PORT = 5005
EEG_PORT = 5006


def level(stress: float) -> str:
    return "calm" if stress < 0.35 else "tense" if stress < 0.65 else "stressed"


class StressSender:
    """Sends API messages over UDP, at most `rate` per second, adding the version and a sequence number."""

    def __init__(self, address: str = f"127.0.0.1:{FACE_PORT}", rate: float = 20.0):
        host, port = address.rsplit(":", 1)
        self.target = (host, int(port))
        self.interval, self._last, self._seq = 1.0 / rate, -math.inf, 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, t: float, message: dict) -> None:
        if t - self._last < self.interval:
            return
        self._last = t
        self._seq += 1
        try:
            self._sock.sendto(json.dumps({"v": VERSION, "seq": self._seq, **message}).encode(), self.target)
        except OSError:
            pass  # nobody is listening yet; that's fine


class StressClient:
    """Receives API messages inside a game. Call update() once per frame; it never blocks."""

    def __init__(self, port: int = FACE_PORT, host: str = "127.0.0.1", timeout: float = 1.0):
        self.timeout = timeout
        self.value = 0.0         # latest stress 0..1; kept while the detector sends null
        self.level = "calm"
        self.face = False        # is the player's face visible right now
        self.calibrating = False
        self.message = None      # the latest full message, for the other fields
        self._seq, self._received = 0, -math.inf
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)

    @property
    def connected(self) -> bool:
        """False when nothing arrived for `timeout` seconds, e.g. the detector isn't running."""
        return time.monotonic() - self._received < self.timeout

    def update(self) -> None:
        """Reads every message that arrived since the last call and keeps the newest."""
        while True:
            try:
                data = self._sock.recv(65535)
            except (BlockingIOError, ConnectionResetError):
                return
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if msg.get("v") != VERSION:
                continue
            if msg.get("seq", 0) <= self._seq and self.connected:
                continue  # an older message that arrived late
            self._seq, self._received, self.message = msg.get("seq", 0), time.monotonic(), msg
            self.face = bool(msg.get("face"))
            self.calibrating = bool(msg.get("calibrating"))
            if msg.get("stress") is not None:
                self.value = float(msg["stress"])
                self.level = msg.get("level") or level(self.value)

    def close(self) -> None:
        self._sock.close()
