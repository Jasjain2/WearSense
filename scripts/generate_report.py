"""Full LOSO evaluation of the MLP and the Random Forest. Writes everything
to results/.

For each of the 30 subjects: hold the subject out as test, use the next
subject as validation (MLP early stopping), train on the other 28. Both
models see the same training subjects and the same standardized features.

Outputs (results/):
    summary.txt                 headline numbers and all analyses in one file
    loso_per_subject.csv        per held-out subject: accuracy, CI, per-activity accuracy
    per_activity_metrics.csv    precision / recall / F1 per activity, both models
    confusion_matrix.png        MLP, aggregated over the 30 folds
    confusion_matrix_rf.png     Random Forest, same
    failure_cases.txt           confusion patterns, worst (activity, subject) pairs,
                                worst subjects with confidence intervals
    loso_predictions.csv        out-of-fold prediction for every window
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.features import N_FEATURES, RAW_DIM, load_features  # noqa: E402

R = config.RESULTS_DIR


def pct(x: float) -> str:
    """Format a fraction as a percentage with one decimal."""
    return f"{100 * x:.1f}%"


def mean_std(xs) -> str:
    """Format values as 'mean% ± std%' (sample std)."""
    return f"{100 * np.mean(xs):.1f}% ± {100 * np.std(xs, ddof=1):.1f}%"


def macro_f1(preds: pd.DataFrame) -> float:
    """Macro F1 over all out-of-fold predictions."""
    return float(ev.per_activity_metrics(preds)["pooled_f1"].mean())


def main() -> None:
    """Run LOSO for both models and write every output to results/."""
    sys.stdout.reconfigure(encoding="utf-8")
    R.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    data = load_features()
    X, y, subject = data["X"], data["y"], data["subject"]
    print(f"Features: {X.shape[0]} windows x {X.shape[1]} features, "
          f"{len(np.unique(subject))} subjects")

    results = {}
    for model in ["mlp", "rf"]:
        print(f"\nLOSO: {ev.MODEL_NAMES[model]}")
        results[model] = ev.run_loso(X, y, subject, model)

    # -- Per-subject ----------------------------------------------------------
    per_subject = {m: ev.per_subject_table(preds, folds)
                   for m, (preds, _, folds) in results.items()}
    ps = per_subject["mlp"].copy()
    rf_ps = per_subject["rf"][["subject", "accuracy", "ci_low", "ci_high"]]
    ps = ps.merge(rf_ps.rename(columns={"accuracy": "rf_accuracy", "ci_low": "rf_ci_low",
                                        "ci_high": "rf_ci_high"}), on="subject")
    ps = ps.rename(columns={"accuracy": "mlp_accuracy", "ci_low": "mlp_ci_low",
                            "ci_high": "mlp_ci_high"})
    ps["mlp_minus_rf"] = ps["mlp_accuracy"] - ps["rf_accuracy"]
    ps.round(4).to_csv(R / "loso_per_subject.csv", index=False)

    # -- Per-activity ---------------------------------------------------------
    per_act = {m: ev.per_activity_metrics(preds) for m, (preds, _, _) in results.items()}
    pd.concat([df.assign(model=ev.MODEL_NAMES[m]) for m, df in per_act.items()]) \
        .round(4).to_csv(R / "per_activity_metrics.csv", index=False)

    # -- Confusion matrices ---------------------------------------------------
    for m, fname in [("mlp", "confusion_matrix.png"), ("rf", "confusion_matrix_rf.png")]:
        preds = results[m][0]
        ev.plot_confusion_matrix(preds["y_true"], preds["y_pred"],
                                 f"{ev.MODEL_NAMES[m]}, LOSO over 30 subjects", R / fname)

    # -- Failure cases ----------------------------------------------------------
    reports = [ev.format_failure_report(results[m][0], per_subject[m],
                                        f"Failure cases: {ev.MODEL_NAMES[m]}")
               for m in ["mlp", "rf"]]
    (R / "failure_cases.txt").write_text("\n\n\n".join(reports) + "\n", encoding="utf-8")

    # -- Predictions --------------------------------------------------------------
    pred_df = results["mlp"][0][["index", "subject", "bout", "y_true"]].copy()
    pred_df["true"] = [ev.ACTIVITY_LABELS[c] for c in pred_df["y_true"]]
    for m, (preds, _, _) in results.items():
        pred_df[f"{m}_pred"] = [ev.ACTIVITY_LABELS[c] for c in preds["y_pred"]]
        pred_df[f"{m}_confidence"] = preds["confidence"].round(4)
    pred_df.drop(columns="y_true").to_csv(R / "loso_predictions.csv", index=False)

    # -- Summary ------------------------------------------------------------------
    lines = summary_lines(results, per_subject, per_act, ps, X.shape[0])
    lines.append(f"\nTotal runtime: {(time.perf_counter() - t_start) / 60:.1f} min")
    text = "\n".join(lines)
    (R / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nAll results written to {R}")


def summary_lines(results, per_subject, per_act, ps, n_windows) -> list[str]:
    """Build summary.txt."""
    L = ["WearSense: LOSO evaluation summary", "=" * 34, ""]
    L += ["Data",
          f"  UCI HAR, 30 subjects, {n_windows} windows of 128 samples x 9 channels",
          f"  Features: {RAW_DIM} raw values -> {N_FEATURES} features per window "
          f"({1 - N_FEATURES / RAW_DIM:.1%} dimensionality reduction)", ""]

    L += ["Leave-one-subject-out accuracy (30 folds)",
          f"  {'Model':<28} {'mean ± std':<16} {'95% CI (subjects)':<18} "
          f"{'pooled':>7} {'macro F1':>9} {'min':>6} {'max':>6}"]
    for m, (preds, _, folds) in results.items():
        acc = folds["accuracy"]
        lo, hi = ev.bootstrap_mean_ci(acc)
        pooled = (preds["y_true"] == preds["y_pred"]).mean()
        L.append(f"  {ev.MODEL_NAMES[m]:<28} {mean_std(acc):<16} {pct(lo) + ' - ' + pct(hi):<18} "
                 f"{pct(pooled):>7} {macro_f1(preds):9.3f} {pct(acc.min()):>6} {pct(acc.max()):>6}")
    diff = ps["mlp_minus_rf"]
    lo, hi = ev.bootstrap_mean_ci(diff)
    L += [f"  MLP - RF per subject: mean {100 * diff.mean():+.1f} pts "
          f"(95% CI {100 * lo:+.1f} to {100 * hi:+.1f}); MLP better on "
          f"{(diff > 0).sum()}/30, RF better on {(diff < 0).sum()}/30, tied {(diff == 0).sum()}",
          "  'pooled' = accuracy over all windows; 'mean' = average of 30 per-subject accuracies."]
    folds = results["mlp"][2]
    L += [f"  MLP early stopping: best epoch median {int(folds['best_epoch'].median())} "
          f"(range {int(folds['best_epoch'].min())}-{int(folds['best_epoch'].max())}), "
          f"epochs run median {int(folds['epochs_run'].median())}", ""]

    for m in ["mlp", "rf"]:
        df = per_act[m]
        L += [f"Per-activity metrics: {ev.MODEL_NAMES[m]} (mean across folds; pooled F1)",
              f"  {'activity':<20} {'n':>5} {'prec':>6} {'recall':>7} {'F1':>6} "
              f"{'F1 sd':>6} {'pooled F1':>10}  top confusion"]
        for r in df.itertuples():
            L.append(f"  {r.activity:<20} {r.n_windows:>5} {r.fold_mean_precision:6.3f} "
                     f"{r.fold_mean_recall:7.3f} {r.fold_mean_f1:6.3f} {r.fold_std_f1:6.3f} "
                     f"{r.pooled_f1:10.3f}  {r.top_confusion}")
        L.append("")

    for m in ["mlp", "rf"]:
        pairs = ev.pair_confusion(results[m][0])
        L.append(f"Expected confusion pairs: {ev.MODEL_NAMES[m]}")
        for r in pairs.itertuples():
            L.append(f"  {r.true:>18} -> {r.predicted:<18} {r.count:>4}  "
                     f"({r.rate:.1%} of {r.true}, {r.share_of_all_errors:.1%} of all errors)")
        L.append(f"  Two pairs combined: {pairs['share_of_all_errors'].sum():.1%} of all errors")
        L.append("")

    for m in ["mlp", "rf"]:
        psm = per_subject[m]
        worst = ev.worst_subjects(psm)
        L += [f"Worst {len(worst)} subjects: {ev.MODEL_NAMES[m]} "
              f"(mean over folds {pct(psm['accuracy'].mean())}; CI = bout bootstrap)",
              f"  {'subj':<5} {'acc':>6}  {'95% CI':<15} {'drop vs mean':>13} "
              f"{'drop vs others':>15}  weakest activity"]
        for r in worst.itertuples():
            L.append(f"  s{r.subject:<4} {pct(r.accuracy):>6}  "
                     f"{pct(r.ci_low) + ' - ' + pct(r.ci_high):<15} "
                     f"{-100 * r.drop_vs_mean:+9.1f} pts {-100 * r.drop_vs_others:+11.1f} pts"
                     f"  {r.weakest_activity} ({pct(r.weakest_activity_acc)})")
        L.append("")

    L += ["Per-subject accuracy (MLP | RF)"]
    for r in ps.sort_values("subject").itertuples():
        L.append(f"  s{r.subject:<3} n={r.n_windows:<4} {pct(r.mlp_accuracy):>6} | "
                 f"{pct(r.rf_accuracy):>6}")
    return L


if __name__ == "__main__":
    main()
