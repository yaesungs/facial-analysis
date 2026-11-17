# Stress API (v1)

How the stress detector talks to games. The detector and the game are separate
programs; this message format is the only thing they share. Either side can be
rewritten (pygame → Unity, webcam → Muse EEG) without touching the other.

## Transport

- **UDP** on the same computer: `127.0.0.1`.
- Port **5005**: face detector (`stress_cam.py`) or the simulator (`stress_simulator.py`).
  Port **5006** is reserved for the Muse 2 EEG.
- One JSON message (UTF-8) per packet, up to 20 per second.
- Every message is the complete current state. The newest message wins, and a lost
  message is simply replaced by the next one.

## Message

```json
{"v": 1, "seq": 1042, "source": "face", "time": 81.42,
 "stress": 0.63, "level": "tense", "face": true, "calibrating": false,
 "heart_rate": 74.0, "blink_rate": 12.0}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | int | API version, currently `1`. Ignore messages with another version. |
| `seq` | int | Counts up by 1 per message. A message with a lower `seq` than the last one arrived late: ignore it. Starts again at 1 when the sender restarts. |
| `source` | string | `"face"`, `"eeg"` or `"simulator"`. |
| `time` | float | Seconds since the sender started. |
| `stress` | float or null | 0.0 (calm) to 1.0 (very stressed). `null` when there is no value yet (while calibrating, or the moment it ends): keep using the last value. |
| `level` | string or null | `"calm"` (below 0.35), `"tense"` (0.35–0.65) or `"stressed"` (above 0.65). |
| `face` | bool | Is the player's face visible. When `false`, `stress` is the last measured value. |
| `calibrating` | bool | The detector is still learning the player's relaxed face. |
| `heart_rate` | float or null | Webcam heart rate in bpm (experimental); `null` when unreliable. |
| `blink_rate` | float or null | Blinks per minute over the last 30 s. |

Receivers must ignore fields they don't know, so new fields can be added without
breaking older games.

## Rules for game code

1. **Never wait for a message.** Read without blocking once per frame and use the newest.
2. **Hold the last value** when `stress` is `null`.
3. **No message for 1 second** means the detector isn't running: fall back to a neutral
   value (for example 0).
4. Smoothing is up to the game. The detector sends the value unsmoothed so it reacts
   quickly; smooth it if a mechanic needs a steadier number.

## Python / pygame

`stress_api.StressClient` does all of the above:

```python
from stress_api import StressClient

stress = StressClient()             # listens on 127.0.0.1:5005
while running:                      # the game loop
    stress.update()                 # never blocks
    if stress.connected:
        jump = 1.0 + 0.5 * stress.value
```

`stress.level`, `stress.face`, `stress.calibrating` and `stress.message` (the full
latest message) are also available.

## Testing without a webcam

`python stress_simulator.py` sends the same messages from a window with a slider.
Use it to build and tune game mechanics; switch to `python stress_cam.py` later and the
game doesn't change.
