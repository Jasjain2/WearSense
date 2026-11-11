"""Feature extraction from raw inertial windows.

Each window ([128, 9] at 50 Hz) is reduced to a fixed 112-dim vector:
    63  per-channel stats   (mean, std, min, max, median, rms, zcr) x 9
    45  per-channel FFT     (top-5 spectral peak magnitudes) x 9
     4  cross-channel       (mean, std of |body_acc| and |body_gyro|)

128 x 9 = 1152 raw values -> 112 features, a 90.3% reduction.
"""

import numpy as np
import scipy.fft
from scipy.signal import find_peaks

from src import config

STAT_NAMES = ["mean", "std", "min", "max", "median", "rms", "zcr"]
N_FFT_PEAKS = 5
CROSS_NAMES = ["acc_mag_mean", "acc_mag_std", "gyro_mag_mean", "gyro_mag_std"]
N_CHANNELS = len(config.CHANNELS)
N_FEATURES = (len(STAT_NAMES) + N_FFT_PEAKS) * N_CHANNELS + len(CROSS_NAMES)
RAW_DIM = config.WINDOW_LEN * N_CHANNELS


def feature_names() -> list[str]:
    """Names in the same order as extract_features output."""
    names = [f"{ch}_{s}" for ch in config.CHANNELS for s in STAT_NAMES]
    names += [f"{ch}_fft_peak{k + 1}" for ch in config.CHANNELS for k in range(N_FFT_PEAKS)]
    return names + CROSS_NAMES


def _stat_features(w: np.ndarray) -> np.ndarray:
    """[9 * 7] channel-major: all stats for total_acc_x, then total_acc_y, ..."""
    n = w.shape[0]
    sign_changes = np.sum(np.diff(np.signbit(w), axis=0), axis=0)
    stats = np.stack([
        w.mean(axis=0),
        w.std(axis=0),
        w.min(axis=0),
        w.max(axis=0),
        np.median(w, axis=0),
        np.sqrt(np.mean(w ** 2, axis=0)),
        sign_changes / n,
    ], axis=1)  # [9, 7]
    return stats.ravel()


def _fft_peak_magnitudes(x: np.ndarray, k: int = N_FFT_PEAKS) -> np.ndarray:
    """Magnitudes of the k largest spectral peaks of a 1D signal, excluding DC.

    Magnitudes are divided by the signal length (raw rfft magnitudes scale
    with N). If fewer than k local peaks exist, the remaining slots fall back
    to the largest non-peak bins, then zeros.
    """
    mag = np.abs(scipy.fft.rfft(x)) / len(x)
    mag = mag[1:]  # drop DC
    peaks, _ = find_peaks(mag)
    top = np.sort(mag[peaks])[::-1][:k]
    if len(top) < k:
        rest = np.sort(np.delete(mag, peaks))[::-1][:k - len(top)]
        top = np.concatenate([top, rest])
    return np.pad(top, (0, k - len(top)))


def _fft_features(w: np.ndarray) -> np.ndarray:
    return np.concatenate([_fft_peak_magnitudes(w[:, c]) for c in range(w.shape[1])])


def _cross_channel_features(w: np.ndarray) -> np.ndarray:
    """Body (gravity-removed) acceleration is used for the magnitude: the
    total-acceleration magnitude is ~1 g in every window, so its mean says
    little about movement."""
    acc_mag = np.linalg.norm(w[:, config.BODY_ACC_IDX], axis=1)
    gyro_mag = np.linalg.norm(w[:, config.GYRO_IDX], axis=1)
    return np.array([acc_mag.mean(), acc_mag.std(), gyro_mag.mean(), gyro_mag.std()])


def extract_features(window: np.ndarray) -> np.ndarray:
    """Map one raw window [128, 9] to a 1D feature vector [112]."""
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected [num_samples, {N_CHANNELS}], got {w.shape}")
    if w.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples, got {w.shape[0]}")
    feats = np.concatenate([
        _stat_features(w),
        _fft_features(w),
        _cross_channel_features(w),
    ])
    assert feats.shape == (N_FEATURES,)
    return feats.astype(np.float32)


def extract_features_batch(X: np.ndarray) -> np.ndarray:
    """[num_windows, 128, 9] -> [num_windows, 112] float32."""
    if len(X) == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    return np.stack([extract_features(w) for w in X])


def load_features(rebuild: bool = False) -> dict[str, np.ndarray]:
    """Features for all 10,299 windows, cached in data/features.npz.

    Returns {"X": [n, 112] float32, "y": [n] int64, "subject": [n] int64}.
    """
    from src.data_loader import load_all

    cache = config.FEATURE_CACHE
    if cache.exists() and not rebuild:
        with np.load(cache) as z:
            if z["X"].shape[1] == N_FEATURES:
                return {k: z[k] for k in ("X", "y", "subject")}
    data = load_all()
    out = {"X": extract_features_batch(data["X"]), "y": data["y"], "subject": data["subject"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    from src.data_loader import load_split

    data = load_split("train")
    window = data["X"][0]
    feats = extract_features(window)
    names = feature_names()

    print(f"Raw window shape: {window.shape}  ({RAW_DIM} values)")
    print(f"Feature shape:    {feats.shape}  (expected ({N_FEATURES},))")
    print(f"Reduction:        {RAW_DIM} -> {N_FEATURES} "
          f"({1 - N_FEATURES / RAW_DIM:.1%} fewer dimensions)")
    for i in [0, 1, 6, 63, 64, 108, 109, 110, 111]:
        print(f"  [{i:3d}] {names[i]:<26} {feats[i]: .5f}")
    assert len(names) == N_FEATURES == 112
    assert len(set(names)) == N_FEATURES, "duplicate feature names"
    assert np.isfinite(feats).all(), "NaN/Inf in features"
    print("OK: no NaN/Inf")
