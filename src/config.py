"""Central configuration: dataset layout, paths, and hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_DIR = DATA_ROOT / "UCI HAR Dataset"
FEATURE_CACHE = DATA_ROOT / "features.npz"   # extracted features for all windows
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    # Current UCI repository link; wraps "UCI HAR Dataset.zip" inside another zip.
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
]
SPLITS = ["train", "test"]  # the original 21/9 subject split; LOSO merges both

# Activity IDs as in activity_labels.txt (1-based).
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
ACTIVITIES = list(ACTIVITY_NAMES)
# Contiguous class indices 0..5 for the classifier.
ACTIVITY_TO_LABEL = {a: i for i, a in enumerate(ACTIVITIES)}
LABEL_TO_ACTIVITY = {i: a for a, i in ACTIVITY_TO_LABEL.items()}
NUM_CLASSES = len(ACTIVITIES)

ALL_SUBJECTS = list(range(1, 31))
EXPECTED_WINDOWS = 10299

# Expected confusion pairs: similar dynamics (stairs) or similar posture (sit/stand).
CONFUSION_PAIRS = [("WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"), ("SITTING", "STANDING")]

# Raw inertial channels, in the order they are stacked. Files are
# Inertial Signals/{channel}_{split}.txt.
CHANNELS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]
BODY_ACC_IDX = [3, 4, 5]
GYRO_IDX = [6, 7, 8]

# ---------------------------------------------------------------------------
# Sensor properties
# ---------------------------------------------------------------------------
SAMPLE_RATE = 50       # Hz
WINDOW_LEN = 128       # samples per window (2.56 s, 50% overlap)

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15  # epochs without val-loss improvement
RF_TREES = 100

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 1000     # bootstrap resamples for confidence intervals
CI_LEVEL = 0.95
N_WORST_SUBJECTS = 5
N_WORST_PAIRS = 10     # (activity, subject) cells listed in the failure report

RANDOM_SEED = 42
