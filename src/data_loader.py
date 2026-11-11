"""Load the raw UCI HAR inertial signals, grouped by subject.

The dataset ships pre-segmented 128-sample windows split into train (21
subjects) and test (9 subjects). For leave-one-subject-out evaluation both
splits are merged and regrouped by subject. Only the raw signals in
Inertial Signals/ are used; the dataset's 561 pre-computed features are not.
"""

from collections import Counter

import numpy as np

from src import config


def _read_matrix(path) -> np.ndarray:
    """Whitespace-separated numeric text file -> float32 array."""
    return np.loadtxt(path, dtype=np.float32, ndmin=2)


def load_split(split: str) -> dict[str, np.ndarray]:
    """Load one original split.

    Returns:
        X:       [num_windows, 128, 9] float32, channels in config.CHANNELS order
        y:       [num_windows] int64 class indices (config.ACTIVITY_TO_LABEL)
        subject: [num_windows] int64 subject IDs (1-30)
    """
    d = config.DATA_DIR / split
    signals = [_read_matrix(d / "Inertial Signals" / f"{ch}_{split}.txt")
               for ch in config.CHANNELS]
    X = np.stack(signals, axis=-1)
    activity = np.loadtxt(d / f"y_{split}.txt", dtype=np.int64)
    subject = np.loadtxt(d / f"subject_{split}.txt", dtype=np.int64)
    if X.shape[1:] != (config.WINDOW_LEN, len(config.CHANNELS)):
        raise ValueError(f"Unexpected window shape {X.shape} in {split}")
    if not len(X) == len(activity) == len(subject):
        raise ValueError(f"Row counts differ in {split}: X {len(X)}, "
                         f"y {len(activity)}, subject {len(subject)}")
    y = np.array([config.ACTIVITY_TO_LABEL[a] for a in activity], dtype=np.int64)
    return {"X": X, "y": y, "subject": subject}


def load_all() -> dict[str, np.ndarray]:
    """Both splits concatenated: X [10299, 128, 9], y, subject."""
    parts = [load_split(s) for s in config.SPLITS]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_by_subject() -> dict[int, dict[str, np.ndarray]]:
    """{subject_id: {"X": [n, 128, 9], "y": [n]}} for all 30 subjects."""
    data = load_all()
    return {int(s): {"X": data["X"][data["subject"] == s], "y": data["y"][data["subject"] == s]}
            for s in np.unique(data["subject"])}


def print_summary(data: dict[str, np.ndarray]) -> None:
    """Print window counts per subject, per activity, and the total."""
    names = [config.ACTIVITY_NAMES[a] for a in config.ACTIVITIES]
    by_subject = Counter(data["subject"].tolist())
    by_label = Counter(data["y"].tolist())

    print("\n=== Windows per subject ===")
    subs = sorted(by_subject)
    for i in range(0, len(subs), 6):
        print("  " + "   ".join(f"s{s:<2} {by_subject[s]:>3}" for s in subs[i:i + 6]))
    counts = np.array([by_subject[s] for s in subs])
    print(f"  min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    print("\n=== Windows per activity ===")
    for c, name in enumerate(names):
        print(f"  {name:<20} {by_label[c]:>5}  ({by_label[c] / len(data['y']):.1%})")

    print("\n=== Totals ===")
    print(f"  subjects: {len(subs)}")
    print(f"  windows:  {len(data['y'])} (expected {config.EXPECTED_WINDOWS})")
    print(f"  window:   {data['X'].shape[1]} samples x {data['X'].shape[2]} channels "
          f"({data['X'].shape[1] / config.SAMPLE_RATE:.2f} s @ {config.SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    print_summary(load_all())
