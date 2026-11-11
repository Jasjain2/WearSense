# WearSense: Human Activity Recognition from Smartphone Inertial Sensors

## Overview

WearSense classifies six daily activities from a waist-worn smartphone's accelerometer and gyroscope, using the UCI HAR dataset (30 subjects, 10,299 windows). I ignore the dataset's 561 pre-computed features and extract my own from the raw signals. Each 2.56 s window of 128 samples × 9 channels (1,152 values) becomes 112 statistical and spectral features, a **90.3% reduction in dimensionality**. A one-hidden-layer PyTorch MLP classifies the features. I evaluate it with **leave-one-subject-out (LOSO) cross-validation across all 30 participants**, so every test subject is unseen in training. The MLP reaches **92.8% ± 5.4%** mean LOSO accuracy, against **91.1% ± 6.5%** for a 100-tree Random Forest on the same features. Most errors (59%) are SITTING/STANDING confusions.

## Key Results

LOSO over 30 subjects. Mean ± sample std of the 30 per-subject accuracies; 95% CI from a bootstrap over subjects.

| Model                     | Features | LOSO accuracy (mean ± std) | 95% CI        | Pooled accuracy | Macro F1 | Worst subject |
|---------------------------|----------|----------------------------|---------------|-----------------|----------|---------------|
| **MLP (128, dp=0.3)**     | 112      | **92.8% ± 5.4%**           | 90.8% - 94.6% | 93.0%           | 0.929    | 77.1% (s14)   |
| Random Forest (100 trees) | 112      | 91.1% ± 6.5%               | 88.8% - 93.3% | 91.2%           | 0.910    | 77.4% (s14)   |

Pooled accuracy is computed over all 10,299 out-of-fold predictions. Paired per subject, the MLP beats the RF by +1.7 points (95% CI +0.3 to +3.2), but the advantage is not uniform: the MLP is better on 15 subjects, the RF on 13, and 2 are tied.

## Dataset

[UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (Human Activity Recognition Using Smartphones) has 30 volunteers aged 19-48. Each performed six activities while wearing a Samsung Galaxy S II on the waist. The phone recorded a 3-axis accelerometer and a 3-axis gyroscope at 50 Hz. The signals were noise-filtered, and the acceleration was separated into gravity and body components. They were then segmented into 2.56 s windows (128 samples) with 50% overlap.

| Activity           | Windows | Activity | Windows |
|--------------------|---------|----------|---------|
| WALKING            | 1,722   | SITTING  | 1,777   |
| WALKING_UPSTAIRS   | 1,544   | STANDING | 1,906   |
| WALKING_DOWNSTAIRS | 1,406   | LAYING   | 1,944   |

That makes 10,299 windows in total, 281-409 per subject. I use the 9 raw channels in `Inertial Signals/`: `total_acc_{x,y,z}` (gravity included), `body_acc_{x,y,z}` and `body_gyro_{x,y,z}`. The dataset's original 21/9 subject train/test split is merged and regrouped by subject for LOSO.

> D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. "A Public Domain Dataset for Human Activity Recognition Using Smartphones." ESANN, 2013.

## Method

```
 raw window [128 samples x 9 channels] = 1,152 values
 total_acc xyz | body_acc xyz | body_gyro xyz   (50 Hz, 2.56 s)
                         |
                         v
 feature extraction (src/features.py)
   per channel: mean, std, min, max, median, RMS, zero-crossing rate   7 x 9 = 63
   per channel: top-5 FFT peak magnitudes (normalized by length)       5 x 9 = 45
   cross-channel: mean/std of |body_acc| and |body_gyro|                       4
                         |
                         v
               112 features (-90.3%)
                         |
                         v
   StandardScaler (fit on the 28 training subjects of each fold)
                         |
                         v
   MLP: Linear(112,128) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(128,6)
```

**Feature extraction** ([src/features.py](src/features.py)). The approach follows MotionFuse's IMU features. For each of the 9 channels I take 7 time-domain statistics and the magnitudes of the 5 largest peaks in the `scipy.fft.rfft` spectrum (DC excluded, divided by window length). Four cross-channel features add the mean and std of the acceleration and gyroscope magnitude vectors. The acceleration magnitude uses `body_acc` rather than `total_acc`, because the total-acceleration magnitude is about 1 g in every window and says little about movement. Gravity orientation, which separates the static postures, is still captured by the per-axis `total_acc` means. `feature_names()` returns all 112 names in order.

**Classification** ([src/models.py](src/models.py), [src/train.py](src/train.py)). `ActivityClassifier` has a single hidden layer of 128 units. In MotionFuse, deeper networks did not beat a small MLP, so I kept it simple. Training uses Adam (lr 1e-3, weight decay 1e-4), cross-entropy loss and batches of 64, for at most 150 epochs. Early stopping halts training after 15 epochs without improvement in validation loss, and the best-validation weights are restored. The Random Forest baseline uses scikit-learn with 100 trees on the same standardized features.

## Evaluation Protocol

- **LOSO cross-validation** ([src/evaluate.py](src/evaluate.py)). There is one fold per subject (30 folds):
  - **test:** the held-out subject;
  - **validation:** the next subject in ID order (s30 wraps to s1), used only for MLP early stopping, so every subject is a validation subject exactly once;
  - **train:** the remaining 28 subjects. Both models train on the same 28 subjects, and the scaler is fit on them alone.
- **Per-activity:** precision, recall and F1 computed within each fold and averaged over the 30 folds, plus pooled values. The expected confusion pairs WALKING_UPSTAIRS/WALKING_DOWNSTAIRS and SITTING/STANDING are tracked separately.
- **Per-subject:** accuracy per held-out subject and per (activity, subject) cell, which shows the subjects and activities the model struggles with.
- **Confusion matrix:** aggregated over all 30 folds.
- **Failure cases:** every confusion pattern with its contributing subjects, the 10 worst (activity, subject) pairs, and the 5 worst subjects.
- **Confidence intervals.** Neighbouring windows share 50% of their samples, so treating windows as independent would make intervals too narrow. Per-subject CIs therefore use a **cluster bootstrap over activity bouts**: contiguous runs of same-label windows, about 12-15 per subject. CIs on the mean LOSO accuracy bootstrap over the 30 subjects.
- **Fixed seeds:** the MLP uses seed 42 in every fold and the RF `random_state=42`, so reruns on the same machine and library versions should reproduce these numbers.

## Key Findings

- **The errors are concentrated in the two expected confusion pairs.** Together the two pairs account for 71.6% of the MLP's 725 errors. The rest are walking-family mix-ups (WALKING vs the stair classes). Moving and static activities are almost never confused with each other: 1 window in 10,299 crosses between the two groups, and LAYING is classified perfectly (1,944 / 1,944).

  | Confusion (MLP)                         | Windows | Rate within true class | Share of all errors |
  |-----------------------------------------|---------|------------------------|---------------------|
  | SITTING → STANDING                      | 236     | 13.3%                  | 32.6%               |
  | STANDING → SITTING                      | 194     | 10.2%                  | 26.8%               |
  | WALKING_UPSTAIRS → WALKING_DOWNSTAIRS   | 54      | 3.5%                   | 7.4%                |
  | WALKING_DOWNSTAIRS → WALKING_UPSTAIRS   | 35      | 2.5%                   | 4.8%                |

- **SITTING vs STANDING is the hardest distinction.** These are the lowest-F1 classes, at 0.866 (SITTING) and 0.879 (STANDING) fold-mean, against 0.924-0.999 for the others. A waist-mounted phone sees nearly the same gravity vector in both postures: the trunk is upright either way, and only the thighs change. Errors in this pair are subject-specific. For example, s28 SITTING is 47.2% (38 of 72 windows called STANDING), and s16 and s25 STANDING are 50.0% and 51.4%. Seven of the 10 worst (activity, subject) pairs are SITTING or STANDING cells, and in every one of them all errors go to the other posture.
- **The stair pair is less of a problem than expected.** The MLP confuses upstairs and downstairs in only 2.5-3.5% of their windows. The exception is subject 6, whose WALKING_UPSTAIRS accuracy is 41.2%: 20 windows are called downstairs and 10 called walking.
- **One subject's walking does not resemble anyone else's.** All 59 of subject 14's WALKING windows are classified as WALKING_UPSTAIRS, with a mean confidence of 0.89. The Random Forest makes the same mistake, so the cause lies in the data rather than in the model. That single cell explains 59 of the MLP's 86 WALKING errors and most of s14's 15.7-point drop.
- **Accuracy drops on the 5 worst-performing subjects (MLP)**, against the 92.8% mean:

  | Subject | Accuracy | 95% CI (bout bootstrap) | Drop vs mean | Weakest activity           |
  |---------|----------|-------------------------|--------------|----------------------------|
  | s14     | 77.1%    | 52.0% - 96.3%           | -15.7 pts    | WALKING (0.0%)             |
  | s16     | 83.6%    | 61.4% - 100.0%          | -9.2 pts     | STANDING (50.0%)           |
  | s9      | 84.4%    | 72.3% - 95.7%           | -8.4 pts     | STANDING (42.2%)           |
  | s7      | 87.0%    | 75.2% - 96.5%           | -5.8 pts     | SITTING (50.0%)            |
  | s10     | 87.1%    | 66.9% - 100.0%          | -5.7 pts     | SITTING (55.6%)            |

  The intervals are wide because each subject contributes only about 12 independent bouts, and one badly classified bout moves accuracy by several points. Four of the five worst subjects fail on a sit/stand posture, not on movement.
- **Per-subject variance dominates.** Subject accuracy ranges from 77.1% to 99.4%, and 11 subjects score at least 96%. The std across subjects (5.4 points) is three times the MLP-vs-RF gap (1.7 points).

## Reproduction

Tested on Windows 11, Python 3.14, CPU only.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download and extract UCI HAR (~60 MB). Only the raw inertial signals, labels and subject IDs are extracted to `data/UCI HAR Dataset/`:
   ```bash
   python setup_data.py
   ```
3. Run the full pipeline: data summary, feature extraction (cached to `data/features.npz`), then LOSO for both models. It takes about 8 min on CPU, roughly 15 s per MLP fold and 1 s per RF fold:
   ```bash
   python scripts/run_pipeline.py            # add --rebuild to re-extract features
   ```
   `python scripts/generate_report.py` reruns only the LOSO evaluation from cached features.
4. Results are written to `results/`:
   - `summary.txt`: headline numbers and every analysis in one file
   - `loso_per_subject.csv`: per-subject MLP and RF accuracy with CIs, per-activity accuracy, early-stopping epoch
   - `per_activity_metrics.csv`: fold-mean and pooled precision/recall/F1 for both models
   - `confusion_matrix.png` (MLP) and `confusion_matrix_rf.png`
   - `failure_cases.txt`: confusion patterns, worst (activity, subject) pairs, worst subjects
   - `loso_predictions.csv`: the out-of-fold prediction and confidence for every window

Individual modules can be checked on their own: `python -m src.data_loader` prints window counts, and `python -m src.features` checks feature shapes and names.

## Repository Structure

```
wearsense/
├── README.md
├── requirements.txt
├── setup_data.py              Download (two mirror URLs), extract and verify UCI HAR
├── src/
│   ├── config.py              Paths, activity names, channels, hyperparameters
│   ├── data_loader.py         Raw Inertial Signals -> [n, 128, 9] per subject; summary
│   ├── features.py            112 features per window; feature_names(); feature cache
│   ├── models.py              PyTorch ActivityClassifier (one hidden layer)
│   ├── train.py               Training loop with early stopping on validation loss
│   └── evaluate.py            LOSO CV, per-activity/per-subject metrics, bout bootstrap,
│                              failure analysis, confusion-matrix plot
├── scripts/
│   ├── run_pipeline.py        Data summary -> features -> LOSO train + evaluate
│   └── generate_report.py     LOSO for MLP and RF; writes every output to results/
├── data/                      (generated) dataset and feature cache
└── results/                   (generated) summary, CSVs, figures
```

## Limitations

- **Hand-crafted window features only.** Each 2.56 s window is summarized independently, so the order of events within a window and the context across windows (such as a stand-to-sit transition) are discarded. The dataset's activities are also performed in a fixed protocol order, which a sequence model over windows could exploit unfairly. I did not use it.
- **One waist sensor limits posture recognition.** SITTING and STANDING differ mainly in thigh angle, which a waist-mounted phone barely sees. This explains most of the remaining error, and better classifiers are unlikely to close the gap without another sensor position.
- **Single-subject validation is noisy.** Early stopping relies on one validation subject per fold. When that subject is atypical, training can stop very early: in the fold that validates on s14, the best epoch was 2. Best epochs range from 1 to 53 (median 18). Validating on several subjects would make stopping more stable.
- **Wide per-subject intervals.** Each subject has only about 12-15 activity bouts, so per-subject and per-(activity, subject) differences are indicative, not significant. I chose the bout bootstrap deliberately: a window-level bootstrap would report much narrower, overconfident intervals.
- **No hyperparameter search.** The architecture and training settings were fixed in advance and not tuned on LOSO results. Tuning them with the same folds would bias the reported accuracy upward, and a nested LOSO would be needed.
- **Controlled data collection.** The 30 subjects are young adults (19-48) following a scripted protocol, with the same phone model worn in the same position. Accuracy on free-living data, other devices or older populations is untested.
