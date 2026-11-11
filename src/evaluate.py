"""Leave-one-subject-out (LOSO) cross-validation and its analyses:
per-activity, per-subject, confusion matrix, failure cases and worst-subject
confidence intervals.

Class labels are the contiguous indices from config.ACTIVITY_TO_LABEL; plots
and tables use activity names. "Accuracy" for an activity is the fraction of
its windows classified correctly (identical to its recall).

Confidence intervals: neighbouring UCI HAR windows overlap by 50%, so windows
are not independent. Per-subject intervals resample activity *bouts*
(contiguous runs of windows with one label) rather than individual windows.
"""

import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.train import predict_proba, train_model  # noqa: E402

ACTIVITY_LABELS = [config.ACTIVITY_NAMES[a] for a in config.ACTIVITIES]  # index = class label
LABEL_OF = {name: i for i, name in enumerate(ACTIVITY_LABELS)}
MODEL_NAMES = {"mlp": "MLP (128, dp=0.3)", "rf": "Random Forest (100 trees)"}

# Plot styling (same palette as MotionFuse), light mode
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------
def activity_bouts(y: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Bout ID per window: a new bout starts whenever the subject or the label
    changes between consecutive rows. Requires the dataset's original row
    order, which is chronological within each subject."""
    starts = np.r_[True, (y[1:] != y[:-1]) | (subject[1:] != subject[:-1])]
    return np.cumsum(starts) - 1


def validation_subject(test_subject: int, subjects: list[int]) -> int:
    """The next subject after the test subject (wrapping around), so every
    subject serves as validation exactly once."""
    i = subjects.index(test_subject)
    return subjects[(i + 1) % len(subjects)]


def train_rf(X: np.ndarray, y: np.ndarray, seed: int = config.RANDOM_SEED) -> RandomForestClassifier:
    """Fit a Random Forest with config.RF_TREES trees and the given seed."""
    return RandomForestClassifier(n_estimators=config.RF_TREES, random_state=seed,
                                  n_jobs=-1).fit(X, y)


def run_loso(X: np.ndarray, y: np.ndarray, subject: np.ndarray, model: str,
             verbose: bool = True) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Leave-one-subject-out CV for model "mlp" or "rf".

    Per fold: the held-out subject is the test set, validation_subject() is
    the validation set (MLP early stopping only), and the remaining 28
    subjects are the training set. Both models train on the same 28
    subjects, and the StandardScaler is fit on those 28 only.

    Returns:
        preds: one row per window (row order of X): index, subject, bout,
               y_true, y_pred, confidence
        proba: [n, num_classes] predicted class probabilities
        folds: one row per fold: subject, val_subject, n_windows, accuracy,
               best_epoch, epochs_run (NaN for RF), seconds
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {list(MODEL_NAMES)}, got {model!r}")
    subjects = sorted(int(s) for s in np.unique(subject))
    proba = np.zeros((len(y), config.NUM_CLASSES), dtype=np.float32)
    folds = []
    for k, s in enumerate(subjects, 1):
        t0 = time.perf_counter()
        v = validation_subject(s, subjects)
        test, val = subject == s, subject == v
        train = ~(test | val)
        scaler = StandardScaler().fit(X[train])
        X_tr, X_val, X_te = (scaler.transform(X[m]) for m in (train, val, test))

        if model == "mlp":
            net, hist = train_model(X_tr, y[train], X_val, y[val],
                                    input_dim=X.shape[1], num_classes=config.NUM_CLASSES)
            proba[test] = predict_proba(net, X_te)
            best_epoch, epochs_run = hist["best_epoch"], hist["epochs_run"]
        else:
            rf = train_rf(X_tr, y[train])
            proba[test] = rf.predict_proba(X_te)
            best_epoch = epochs_run = np.nan

        acc = float((proba[test].argmax(1) == y[test]).mean())
        secs = time.perf_counter() - t0
        folds.append({"subject": s, "val_subject": v, "n_windows": int(test.sum()),
                      "accuracy": acc, "best_epoch": best_epoch,
                      "epochs_run": epochs_run, "seconds": secs})
        if verbose:
            extra = f"  best epoch {best_epoch:>3}/{epochs_run:<3}" if model == "mlp" else ""
            print(f"  [{model}] fold {k:>2}/{len(subjects)}  test s{s:<2} val s{v:<2} "
                  f"acc {acc:6.1%}{extra}  ({secs:.1f}s)", flush=True)

    y_pred = proba.argmax(1)
    preds = pd.DataFrame({"index": np.arange(len(y)), "subject": subject,
                          "bout": activity_bouts(y, subject), "y_true": y,
                          "y_pred": y_pred, "confidence": proba.max(1)})
    return preds, proba, pd.DataFrame(folds)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(values: np.ndarray, n_boot: int = config.N_BOOTSTRAP,
                      level: float = config.CI_LEVEL,
                      seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of independent values (e.g. the
    30 per-fold accuracies)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def bout_bootstrap_ci(correct: np.ndarray, bouts: np.ndarray, n_boot: int = config.N_BOOTSTRAP,
                      level: float = config.CI_LEVEL,
                      seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """Cluster bootstrap CI for accuracy: resample whole bouts with
    replacement, then pool their windows. Accounts for overlapping windows
    inside a bout being correlated."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(bouts, return_inverse=True)
    n_correct = np.bincount(inv, weights=np.asarray(correct, dtype=float))
    n_total = np.bincount(inv).astype(float)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    accs = n_correct[draws].sum(axis=1) / n_total[draws].sum(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(accs, alpha)), float(np.quantile(accs, 1 - alpha))


# ---------------------------------------------------------------------------
# Per-activity metrics
# ---------------------------------------------------------------------------
def top_confusion(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> str:
    """Most common wrong prediction for class c, e.g. 'STANDING (123)', or ''."""
    wrong = Counter(y_pred[(y_true == c) & (y_pred != c)].tolist())
    if not wrong:
        return ""
    label, n = wrong.most_common(1)[0]
    return f"{ACTIVITY_LABELS[label]} ({n})"


def per_activity_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall and F1 per activity.

    fold_mean_* : computed within each LOSO fold, then averaged over the 30
                  folds (precision is skipped for a fold that never predicts
                  that activity).
    pooled_*    : computed once over all 10,299 out-of-fold predictions.
    """
    labels = list(range(config.NUM_CLASSES))
    per_fold = []
    for _, g in preds.groupby("subject"):
        p, r, f, _ = precision_recall_fscore_support(g["y_true"], g["y_pred"], labels=labels,
                                                     zero_division=np.nan)
        per_fold.append(np.stack([p, r, f]))
    per_fold = np.stack(per_fold)  # [folds, 3, classes]
    mean = np.nanmean(per_fold, axis=0)
    std = np.nanstd(per_fold, axis=0, ddof=1)
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    pp, pr, pf, n = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return pd.DataFrame({
        "activity": ACTIVITY_LABELS, "n_windows": n,
        "fold_mean_precision": mean[0], "fold_mean_recall": mean[1], "fold_mean_f1": mean[2],
        "fold_std_f1": std[2],
        "pooled_precision": pp, "pooled_recall": pr, "pooled_f1": pf,
        "top_confusion": [top_confusion(yt, yp, c) for c in labels],
    })


def pair_confusion(preds: pd.DataFrame) -> pd.DataFrame:
    """For each expected confusion pair (A, B): windows of A predicted as B and
    vice versa, as counts and as a share of that activity's errors."""
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    n_errors = int((yt != yp).sum())
    rows = []
    for a_name, b_name in config.CONFUSION_PAIRS:
        a, b = LABEL_OF[a_name], LABEL_OF[b_name]
        for src, dst in [(a, b), (b, a)]:
            errs = (yt == src) & (yp != src)
            n_to = int(((yt == src) & (yp == dst)).sum())
            rows.append({"pair": f"{a_name} / {b_name}",
                         "true": ACTIVITY_LABELS[src], "predicted": ACTIVITY_LABELS[dst],
                         "count": n_to, "n_true": int((yt == src).sum()),
                         "rate": n_to / (yt == src).sum(),
                         "share_of_activity_errors": n_to / errs.sum() if errs.any() else 0.0,
                         "share_of_all_errors": n_to / n_errors if n_errors else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-subject analysis
# ---------------------------------------------------------------------------
def per_subject_table(preds: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """One row per held-out subject: accuracy with bout-bootstrap CI and the
    accuracy for each activity."""
    rows = []
    for s, g in preds.groupby("subject"):
        correct = (g["y_true"] == g["y_pred"]).to_numpy()
        lo, hi = bout_bootstrap_ci(correct, g["bout"].to_numpy())
        row = {"subject": int(s), "n_windows": len(g), "n_bouts": g["bout"].nunique(),
               "accuracy": correct.mean(), "ci_low": lo, "ci_high": hi}
        for c, name in enumerate(ACTIVITY_LABELS):
            m = (g["y_true"] == c).to_numpy()
            row[f"acc_{name}"] = correct[m].mean() if m.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.merge(folds[["subject", "val_subject", "best_epoch", "epochs_run"]], on="subject")


def worst_subjects(per_subject: pd.DataFrame, k: int = config.N_WORST_SUBJECTS) -> pd.DataFrame:
    """The k lowest-accuracy subjects, with the drop (in accuracy points)
    against the mean over all folds and against the mean of the other folds,
    and each subject's weakest activity."""
    mean_all = per_subject["accuracy"].mean()
    n = len(per_subject)
    worst = per_subject.nsmallest(k, "accuracy").copy()
    worst["drop_vs_mean"] = mean_all - worst["accuracy"]
    worst["drop_vs_others"] = (mean_all * n - worst["accuracy"]) / (n - 1) - worst["accuracy"]
    acc_cols = [f"acc_{a}" for a in ACTIVITY_LABELS]
    worst["weakest_activity"] = worst[acc_cols].idxmin(axis=1).str.removeprefix("acc_")
    worst["weakest_activity_acc"] = worst[acc_cols].min(axis=1)
    return worst.reset_index(drop=True)


def worst_activity_subject_pairs(preds: pd.DataFrame,
                                 k: int = config.N_WORST_PAIRS) -> pd.DataFrame:
    """Lowest-accuracy (activity, subject) cells, with the dominant confusion."""
    rows = []
    for (s, c), g in preds.groupby(["subject", "y_true"]):
        wrong = g.loc[g["y_pred"] != c, "y_pred"]
        top = Counter(wrong.tolist()).most_common(1)
        rows.append({"subject": int(s), "activity": ACTIVITY_LABELS[c], "n_windows": len(g),
                     "accuracy": (g["y_pred"] == c).mean(), "n_wrong": len(wrong),
                     "predicted_as": ACTIVITY_LABELS[top[0][0]] if top else "",
                     "predicted_as_share": top[0][1] / len(wrong) if top else 0.0})
    df = pd.DataFrame(rows).sort_values(["accuracy", "n_windows"], ascending=[True, False])
    return df.head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Failure report
# ---------------------------------------------------------------------------
def confusion_patterns(preds: pd.DataFrame) -> pd.DataFrame:
    """Every (true -> predicted) error type, most frequent first, with the
    subjects that contribute most and the mean confidence of those errors."""
    err = preds[preds["y_true"] != preds["y_pred"]]
    rows = []
    for (t, p), g in err.groupby(["y_true", "y_pred"]):
        by_sub = g["subject"].value_counts()
        top = ", ".join(f"s{s} ({n})" for s, n in by_sub.head(3).items())
        rows.append({"true": ACTIVITY_LABELS[t], "predicted": ACTIVITY_LABELS[p],
                     "count": len(g), "share_of_errors": len(g) / len(err),
                     "n_subjects": by_sub.size, "top_subjects": top,
                     "mean_confidence": g["confidence"].mean()})
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def format_failure_report(preds: pd.DataFrame, per_subject: pd.DataFrame, title: str) -> str:
    """Plain-text failure analysis: confusion patterns, expected pairs, worst
    (activity, subject) cells, worst subjects."""
    n, n_err = len(preds), int((preds["y_true"] != preds["y_pred"]).sum())
    lines = [title, "=" * len(title),
             f"{n_err} of {n} windows misclassified across 30 LOSO folds ({n_err / n:.1%})", ""]

    pairs = pair_confusion(preds)
    lines += ["Expected confusion pairs", "-" * 24]
    for pair, g in pairs.groupby("pair", sort=False):
        total = g["count"].sum()
        lines.append(f"  {pair}: {total} errors ({g['share_of_all_errors'].sum():.1%} of all errors)")
        for r in g.itertuples():
            lines.append(f"    {r.true:>18} -> {r.predicted:<18} {r.count:>4} "
                         f"({r.rate:.1%} of {r.true} windows, "
                         f"{r.share_of_activity_errors:.0%} of its errors)")
    lines.append(f"  Both pairs together: {pairs['share_of_all_errors'].sum():.1%} of all errors")
    lines.append("")

    pats = confusion_patterns(preds)
    lines += ["All confusion patterns (true -> predicted)", "-" * 42]
    for r in pats.itertuples():
        lines.append(f"  {r.true:>18} -> {r.predicted:<18} {r.count:>4} ({r.share_of_errors:5.1%})"
                     f"  {r.n_subjects:>2} subjects, top: {r.top_subjects}"
                     f"  mean conf {r.mean_confidence:.2f}")
    lines.append("")

    cells = worst_activity_subject_pairs(preds)
    lines += [f"Worst {len(cells)} (activity, subject) pairs", "-" * 34]
    for r in cells.itertuples():
        conf = (f"-> {r.predicted_as} ({r.predicted_as_share:.0%} of its errors)"
                if r.n_wrong else "")
        lines.append(f"  s{r.subject:<2} {r.activity:<18} {r.accuracy:6.1%}  "
                     f"({r.n_wrong}/{r.n_windows} wrong) {conf}")
    lines.append("")

    worst = worst_subjects(per_subject)
    lo, hi = bootstrap_mean_ci(per_subject["accuracy"])
    lines += [f"Worst {len(worst)} subjects (mean over 30 folds "
              f"{per_subject['accuracy'].mean():.1%}, 95% CI {lo:.1%}-{hi:.1%})",
              "-" * 40,
              f"  {'subj':<5} {'acc':>6}  {'95% CI (bouts)':<15} {'drop':>9}  weakest activity"]
    for r in worst.itertuples():
        lines.append(f"  s{r.subject:<4} {r.accuracy:6.1%}  {r.ci_low:5.1%}-{r.ci_high:<8.1%} "
                     f"{-100 * r.drop_vs_mean:+6.1f} pts  {r.weakest_activity} "
                     f"({r.weakest_activity_acc:.1%})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style_axes(ax) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, path) -> None:
    """Confusion-matrix heatmap aggregated over all folds. Cells are coloured
    by row percentage (recall) and annotated with count and percentage."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n{row_pct[i, j]:.1%}" if cm[i, j] else "0"
                       for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(8.4, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    sns.heatmap(100 * row_pct, annot=annot, fmt="", cmap=SEQ_BLUE, vmin=0, vmax=100,
                square=True, linewidths=2, linecolor=SURFACE, cbar_kws={"shrink": 0.75},
                xticklabels=ACTIVITY_LABELS, yticklabels=ACTIVITY_LABELS, ax=ax,
                annot_kws={"fontsize": 9})
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}  (accuracy {acc:.1%}, n={cm.sum()})", color=TEXT, fontsize=12,
                 loc="left", pad=12)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    _style_axes(ax)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT_2, labelsize=8)
    cbar.outline.set_visible(False)
    cbar.set_label("% of true-activity windows", color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
