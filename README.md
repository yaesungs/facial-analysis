# Face stress detector

Part of the **stress-test-gaming-team** project: games that react to how stressed the
player is. This is the webcam half. It reads your face live, estimates stress from
0 (calm) to 1 (very stressed), and sends that number to the game. The Muse 2 EEG can
plug into the same pipeline later.

## Why MediaPipe and not YOLOv8

YOLOv8 is an object detector: it finds *where* a face is, but says nothing about the
expression. You would still need a second model trained on an emotion dataset, and
those datasets label posed expressions (happy, angry, ...), while a stressed or
focused gamer mostly looks "neutral" to them.

MediaPipe Face Landmarker (Google) runs in real time on a laptop CPU and gives, for
every frame, 478 face points plus 52 facial muscle-movement scores ("blendshapes":
brow lowered, lips pressed, eyes blinking, ...). Those correspond to the facial Action
Units that stress research measures, so we use them directly and put the ML where it
matters: a small model trained on **your** calm vs. stressed recordings, because
everyone's stress face is different.

(YOLO would be the right tool for finding many people far away from a camera.)

## Setup (Windows)

Tested with Python 3.11 on Windows 11 (MediaPipe 1.2, OpenCV 5). If `pip` can't find
`mediapipe`, your Python is too new for it: install Python 3.11 or 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The face model (~3.7 MB) downloads into `models/` the first time you run the app.

## 1. Try it

```bash
python stress_cam.py
```

- First 20 s: calibration. Sit relaxed and look at the screen the way you normally do.
- Then you get a live stress % (calm / tense / stressed), what caused it ("furrowed
  brow, pressed lips"), your blink rate, and an experimental webcam heart rate.
- Keys: `q` quit, `c` recalibrate, `d` show every feature.
- Options: `--camera 1` (another webcam), `--no-window` (console only, e.g. while a game
  runs full screen), `--sensitivity 0.6` (react more), `--no-pulse`, `--video clip.mp4`.
  `python stress_cam.py run -h` lists them all.

## 2. Train your personal model (recommended)

Out of the box, the score uses general rules ("a furrowed brow means stress"). To learn
*your* tells, record labeled sessions while you actually play:

```bash
python stress_cam.py record calm
```

```bash
python stress_cam.py record stressed
```

Each recording is 2 minutes (`--seconds` to change) after a 10 s countdown, so you can
switch to the game first. Record each label at least twice, ideally on different days,
then:

```bash
python stress_cam.py train
```

`train` reports the accuracy on sessions it did not train on (the honest number) and
prints which cues give you away, e.g. `pressed lips  higher when stressed`. From then on
`python stress_cam.py` uses your model automatically (`--baseline` switches back).

Getting good data:
- Keep calm and stressed sessions alike in everything except the stress: same game,
  seat, lighting, and time of day, alternating the order. Otherwise the model learns
  "lamp on vs. off" instead of stress.
- Calm: easy levels or free roaming. Stressed: a level you keep dying on, a horror game,
  or counting down from 1000 in steps of 7, out loud, as fast as you can.

## 3. Use it in a game

While running, up to 20 JSON messages per second go to UDP `127.0.0.1:5005`:

```json
{"stress": 0.63, "level": "tense", "face": true, "calibrating": false,
 "heart_rate": 74.0, "blink_rate": 12.0, "time": 81.4}
```

`stress` is `null` while calibrating. `face` is `false` when the player looks away (the
last stress value is kept). Any engine can read UDP (Unity `UdpClient`, Godot
`PacketPeerUDP`, Python `socket`). To see it work, run this in a second terminal while
the detector runs:

```bash
python game_receiver_example.py
```

## How the score works

| Signal | Comes from | Meaning |
|---|---|---|
| furrowed brow, worried brow, squinting, wide eyes, nose wrinkle | brow and eye blendshapes | effort, frustration, alarm |
| pressed or sucked-in lips, mouth corners down, stretched lips, raised chin | mouth blendshapes | tension, fear |
| smiling | mouth blendshapes | lowers the score |
| blink rate | blink blendshapes over 30 s | up with anxiety, down with intense focus |
| head movement, leaning in | landmark positions | fidgeting, tension |
| heart rate (beta) | tiny color changes of the cheeks (rPPG) | rises under stress |

### What the percentage means

**Baseline score (no training).** The 20 s calibration records your relaxed average for
each signal and how much it naturally wiggles (its standard deviation, with a minimum
so a very still calibration doesn't make it oversensitive). After that, every frame:

1. Each signal, averaged over the last half second, is measured in "wiggles" away from
   your relaxed average.
2. The first wiggle is ignored, so normal fidgeting doesn't count. Anything beyond it,
   capped at 4, is multiplied by the signal's weight and added up as points. Weights are
   in `WEIGHTS` in `stress_model.py`: furrowed brow 1.0, pressed lips 0.7, worried brow
   and mouth corners down 0.4, most others 0.2–0.3, smiling −0.6.
3. Points become the percentage: `stress = 1 − e^(−0.4 × points)`, where 0.4 is
   `--sensitivity`.

| Points | Stress | Label | Example |
|---|---|---|---|
| 0 | 0% | calm | face looks like your calibration |
| 1 | 33% | calm | brow furrowed 2 wiggles past your normal |
| 2 | 55% | tense | brow furrowed 3 wiggles past your normal |
| 4 | 80% | stressed | brow furrowed 5 or more wiggles past your normal |
| 6 | 91% | stressed | strong brow furrow plus pressed lips |

Under 35% is labeled calm, 35–65% tense, above 65% stressed. So the percentage is not a
probability or a medical measure: it is how far your face has moved from your own
relaxed face, in the directions linked to stress. The weights are starting guesses
based on stress research, not tuned on real players yet. Blink rate is left out because
its direction differs between people (up with anxiety, down with intense focus).

**Personal model.** The percentage is the model's probability that the last half second
looks like your "stressed" recordings rather than your "calm" ones. 50% means it can't
tell.

**Speed.** Facial signals show up in the score within about a quarter of a second. Blink
rate and heart rate are rates, so they need 10–30 s of data and change slowly. The value
is sent without smoothing. If a game mechanic needs a steadier value, smooth it in the
game, or start the detector with `--smoothing 1` (seconds).

## Limitations

- It measures visible tension compared with *your own* baseline, not a medical stress level.
- Poor lighting, glare on glasses, or a face partly out of frame make it worse.
- Some people barely show stress on their face. That is what the EEG is for.
- Webcam heart rate is easily disturbed by movement and flashing screen light. It shows
  `--` whenever the signal is too weak to trust.

## Files

| File | Purpose |
|---|---|
| `stress_cam.py` | the app: webcam loop, window, recording, UDP output |
| `face_features.py` | MediaPipe face landmarks → features (expressions, blinks, head motion) |
| `pulse.py` | experimental webcam heart rate |
| `stress_model.py` | baseline score, personal ML model, training |
| `game_receiver_example.py` | how a game reads the stress value |

`models/` and `recordings/` are created on first use and are git-ignored, because
recordings are personal data.

## Adding the Muse 2 later

Stream the Muse with `muselsl` (or the Mind Monitor app over OSC), compute band powers
(alpha, beta, theta per channel) every second, and add them to the feature dict in
`FeatureExtractor.features()`. Recording and training work unchanged, and the personal
model then learns from face and brain signals together.
