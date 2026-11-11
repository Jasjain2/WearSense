"""Download and extract the UCI HAR dataset.

Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Usage:
    python setup_data.py                    # download, extract, verify
    python setup_data.py --zip path.zip     # use an already-downloaded zip
    python setup_data.py --verify-only      # just check the files

Result:
    data/UCI HAR Dataset/{train,test}/Inertial Signals/*.txt
    data/UCI HAR Dataset/{train,test}/{subject,y}_{split}.txt
"""

import argparse
import io
import logging
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("setup_data")

ROOT_NAME = config.DATA_DIR.name  # "UCI HAR Dataset"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream url to dest. The UCI server doesn't always send Content-Length,
    so the progress bar may show bytes only."""
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while block := r.read(chunk):
                f.write(block)
                bar.update(len(block))
    tmp.replace(dest)


def _open_dataset_zip(zip_path: Path) -> zipfile.ZipFile:
    """Return the zip that holds "UCI HAR Dataset/". The current UCI link wraps
    it in an outer archive, so look one level down if needed."""
    zf = zipfile.ZipFile(zip_path)
    if any(n.startswith(f"{ROOT_NAME}/") for n in zf.namelist()):
        return zf
    inner = [n for n in zf.namelist() if n.endswith(f"{ROOT_NAME}.zip")]
    if not inner:
        raise zipfile.BadZipFile(f"no '{ROOT_NAME}/' folder in {zip_path}")
    return zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))


def extract(zip_path: Path) -> int:
    """Extract only the files the project uses (raw inertial signals, subject
    IDs, labels, activity names) into data/. Returns files written."""
    wanted = ("Inertial Signals/", "subject_", "y_", "activity_labels.txt",
              "README.txt")
    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    root = config.DATA_ROOT.resolve()
    n = 0
    with _open_dataset_zip(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and m.filename.startswith(f"{ROOT_NAME}/")
                   and "__MACOSX" not in m.filename
                   and any(w in m.filename for w in wanted)]
        for m in tqdm(members, desc="Extracting"):
            out = (config.DATA_ROOT / m.filename).resolve()
            if root not in out.parents:  # refuse paths escaping data/
                raise ValueError(f"unsafe path in archive: {m.filename}")
            if out.exists() and out.stat().st_size == m.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(m))
            n += 1
    log.info("Extracted %d new files into %s", n, config.DATA_DIR)
    return n


def expected_files() -> list[Path]:
    """Every file data_loader reads."""
    files = []
    for split in config.SPLITS:
        d = config.DATA_DIR / split
        files += [d / f"subject_{split}.txt", d / f"y_{split}.txt"]
        files += [d / "Inertial Signals" / f"{ch}_{split}.txt" for ch in config.CHANNELS]
    return files


def verify() -> bool:
    """Check that all files exist and the window count matches the dataset."""
    missing = [f for f in expected_files() if not f.exists()]
    for f in missing:
        log.warning("missing %s", f.relative_to(config.PROJECT_ROOT))
    if missing:
        return False
    n = sum(len((config.DATA_DIR / s / f"y_{s}.txt").read_text().split())
            for s in config.SPLITS)
    log.info("All %d files present; %d windows (expected %d)",
             len(expected_files()), n, config.EXPECTED_WINDOWS)
    return n == config.EXPECTED_WINDOWS


def main() -> int:
    """Download, extract and verify the dataset; return a process exit code."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, default=None, help="Pre-downloaded dataset zip")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    if args.verify_only:
        return 0 if verify() else 1
    if not args.zip and all(f.exists() for f in expected_files()) and verify():
        log.info("Dataset already present, skipping download")
        return 0

    if args.zip:
        extract(args.zip.expanduser())
        return 0 if verify() else 1

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "uci_har.zip"
        for url in config.DATASET_URLS:
            try:
                download(url, zip_path)
                extract(zip_path)
                break
            except (OSError, zipfile.BadZipFile) as e:
                log.warning("Failed with %s: %s", url, e)
        else:
            log.error("All download links failed. Download the zip manually from the "
                      "dataset page and rerun with --zip <file>.")
            return 1
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
