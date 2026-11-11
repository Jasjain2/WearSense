"""End-to-end pipeline: load data -> extract features -> LOSO training and
evaluation (MLP and Random Forest) -> results/.

Requires the dataset in data/ (python setup_data.py).

    python scripts/run_pipeline.py              # reuse data/features.npz if present
    python scripts/run_pipeline.py --rebuild    # re-extract features from raw signals
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_report  # noqa: E402

from src.data_loader import load_all, print_summary  # noqa: E402
from src.features import N_FEATURES, RAW_DIM, load_features  # noqa: E402


def banner(name: str) -> None:
    """Print a stage header."""
    print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")


def main() -> None:
    """Run every pipeline stage in order."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true", help="Re-extract features")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    banner("data")
    print_summary(load_all())

    banner("feature extraction")
    t0 = time.perf_counter()
    feats = load_features(rebuild=args.rebuild)
    print(f"  {feats['X'].shape[0]} windows: {RAW_DIM} raw values -> {N_FEATURES} features "
          f"({1 - N_FEATURES / RAW_DIM:.1%} reduction) in {time.perf_counter() - t0:.1f}s")

    banner("LOSO training + evaluation")
    generate_report.main()


if __name__ == "__main__":
    main()
