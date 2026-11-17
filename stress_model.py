"""Feature vector -> stress score between 0 (calm) and 1 (very stressed).

Both scorers have the same `score(features) -> (stress, reasons)` method:

BaselineScorer  Works out of the box. Learns what your relaxed face looks like
                during a short calibration, then measures how far you move away
                from it in the directions stress research points to (furrowed
                brow, pressed lips, ...).

PersonalModel   Machine learning on your own data: a logistic regression trained
                on recordings of you playing calm vs. stressed (see `train`). It
                learns your personal stress face, including cues whose direction
                differs between people, such as blink rate.
"""

from __future__ import annotations

import csv
import math
import time
import warnings
from pathlib import Path

import numpy as np

from face_features import FEATURE_NAMES

LABELS = ("calm", "stressed")

READABLE = {
    "brow_down": "furrowed brow", "brow_inner_up": "worried brow", "eye_squint": "squinting",
    "eye_wide": "wide eyes", "nose_sneer": "nose wrinkle", "lip_press": "pressed lips",
    "lip_roll": "lips sucked in", "mouth_frown": "mouth corners down",
    "mouth_stretch": "stretched lips", "chin_raise": "raised chin", "jaw_open": "open mouth",
    "smile": "smiling", "blink_rate": "blink rate", "head_motion": "head movement",
    "face_size": "leaning in", "heart_rate": "heart rate",
}

# How strongly each feature pushes the baseline score up (+) or down (-).
# blink_rate and jaw_open are left to the personal model on purpose: blinking goes
# up with anxiety but down with intense focus, and an open mouth can mean talking,
# yelling or surprise.
WEIGHTS = {
    "brow_down": 1.0,
    "lip_press": 0.7,
    "brow_inner_up": 0.4,
    "mouth_frown": 0.4,
    "lip_roll": 0.3,
    "eye_squint": 0.3,
    "eye_wide": 0.3,
    "nose_sneer": 0.3,
    "mouth_stretch": 0.3,
    "chin_raise": 0.2,
    "head_motion": 0.3,
    "face_size": 0.2,
    "heart_rate": 0.3,
    "smile": -0.6,
}
# Smallest spread trusted when comparing with the baseline (blendshapes use 0.04),
# so a very still calibration doesn't turn tiny changes into big ones.
MIN_STD = {"blink_rate": 3.0, "head_motion": 0.05, "face_size": 0.01, "heart_rate": 4.0}


class BaselineScorer:
    """No training needed: compares you with your own relaxed baseline."""

    name = "baseline"

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

    def calibrate(self, t: float, feats: dict) -> None:
        """Feed features while the player is relaxed; `ready` turns True when done."""
        if self._started is None:
            self._started = t
        self._samples.append([feats[k] for k in FEATURE_NAMES])
        if t - self._started >= self.calibration_seconds and len(self._samples) >= 30:
            data = np.array(self._samples)
            with warnings.catch_warnings():  # features never measured (e.g. no pulse) stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                self.mean = dict(zip(FEATURE_NAMES, np.nanmean(data, axis=0)))
                self.std = dict(zip(FEATURE_NAMES, np.nanstd(data, axis=0)))

    def score(self, feats: dict) -> tuple[float, dict]:
        total, reasons = 0.0, {}
        for name, weight in WEIGHTS.items():
            x, mean = feats[name], self.mean[name]
            if math.isnan(x) or math.isnan(mean):
                continue
            z = (x - mean) / max(self.std[name], MIN_STD.get(name, 0.04))
            push = weight * min(max(z - 1.0, 0.0), 4.0)  # ignore wiggles within your normal range
            if push:
                reasons[name] = push
                total += push
        return 1.0 - math.exp(-self.sensitivity * max(total, 0.0)), reasons


class PersonalModel:
    """Your trained model (see `train`); needs no calibration."""

    name = "personal model"
    ready = True

    def __init__(self, path: Path):
        import joblib

        saved = joblib.load(path)
        self.pipeline, self.features = saved["pipeline"], saved["features"]

    def score(self, feats: dict) -> tuple[float, dict]:
        x = np.array([[feats.get(k, math.nan) for k in self.features]])
        stress = float(self.pipeline.predict_proba(x)[0, 1])
        # How hard each feature pushes towards "stressed" (in log-odds), to show why.
        pushes = self.pipeline[-1].coef_[0] * self.pipeline[:-1].transform(x)[0]
        return stress, dict(zip(self.features, pushes.tolist()))


class Smoother:
    """Exponential moving average with a time constant in seconds (frame-rate independent)."""

    def __init__(self, seconds: float = 1.0):
        self.seconds = seconds
        self.reset()

    def reset(self) -> None:
        self.value, self._t = None, None

    def update(self, t: float, x: float) -> float:
        if self.value is None or self.seconds <= 0:
            self.value = x
        else:
            self.value += (1.0 - math.exp(-(t - self._t) / self.seconds)) * (x - self.value)
        self._t = t
        return self.value


class Recorder:
    """Writes a labeled session to recordings/<label>_<date-time>.csv, 5 rows per second."""

    def __init__(self, folder: Path, label: str, rate: float = 5.0):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        self.label, self.rows = label, 0
        self._interval, self._last = 1.0 / rate, -math.inf
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["time", "label", *FEATURE_NAMES])

    def write(self, t: float, feats: dict) -> None:
        if t - self._last < self._interval:
            return
        self._last = t
        values = ("" if math.isnan(feats[k]) else f"{feats[k]:.5g}" for k in FEATURE_NAMES)
        self._writer.writerow([f"{t:.2f}", self.label, *values])
        self.rows += 1

    def close(self) -> None:
        self._file.close()
        if self.rows == 0:
            self.path.unlink()


def _read_session(path: Path) -> tuple[str, np.ndarray] | None:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or rows[0]["label"] not in LABELS:
        return None
    x = np.array([[float(r[k]) if r.get(k) else math.nan for k in FEATURE_NAMES] for r in rows])
    return rows[0]["label"], x


def train(recordings: Path, model_path: Path) -> None:
    """Trains the personal model on every session in `recordings` and saves it."""
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sessions = []
    for path in sorted(recordings.glob("*.csv")):
        if loaded := _read_session(path):
            sessions.append((path.name, *loaded))
    count = {label: sum(s[1] == label for s in sessions) for label in LABELS}
    print(f"Found {len(sessions)} sessions: {count['calm']} calm, {count['stressed']} stressed.")
    if min(count.values()) == 0:
        raise SystemExit("Record at least one calm and one stressed session first (see README).")

    X = np.concatenate([s[2] for s in sessions])
    y = np.concatenate([np.full(len(s[2]), LABELS.index(s[1])) for s in sessions])
    groups = np.concatenate([np.full(len(s[2]), i) for i, s in enumerate(sessions)])

    # Only use features measured most of the time (e.g. skip heart_rate if the pulse was off).
    keep = np.isfinite(X).mean(axis=0) >= 0.5
    features = [name for name, kept in zip(FEATURE_NAMES, keep) if kept]
    X = X[:, keep]

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000),
    )

    if min(count.values()) >= 2:
        # Honest accuracy: score every session with a model trained WITHOUT that session.
        # (Rows of one session are near-duplicates, so a random split would look far too good.)
        proba = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut(),
                                  method="predict_proba")[:, 1]
        print(f"\nAccuracy on unseen sessions: {balanced_accuracy_score(y, proba >= 0.5):.0%} "
              f"(ROC AUC {roc_auc_score(y, proba):.2f}; 50% / 0.5 = guessing)")
        for i, (name, label, _) in enumerate(sessions):
            print(f"  {name:<34} {label:<9} predicted stress {proba[groups == i].mean():.0%}")
    else:
        print("\nRecord at least 2 calm and 2 stressed sessions to measure accuracy on unseen sessions.")

    model.fit(X, y)
    coefs = model[-1].coef_[0]
    print("\nWhat gives YOUR stress away (strongest first):")
    for i in np.argsort(-np.abs(coefs))[:6]:
        direction = "higher" if coefs[i] > 0 else "lower"
        print(f"  {READABLE[features[i]]:<20} {direction} when stressed  ({coefs[i]:+.2f})")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": features, "sessions": [s[0] for s in sessions],
                 "trained": time.strftime("%Y-%m-%d %H:%M")}, model_path)
    print(f"\nSaved {model_path.name}. `python stress_cam.py` now uses it automatically.")
