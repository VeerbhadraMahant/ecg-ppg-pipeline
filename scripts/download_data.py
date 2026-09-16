"""Download the two datasets described in datasets.md.

Usage:
    python scripts/download_data.py --dataset challenge2015
    python scripts/download_data.py --dataset mimic_perform_af
    python scripts/download_data.py --dataset all
"""
import argparse
import io
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# Note: the www.physionet.org/content/... URL from datasets.md serves an HTML
# landing page, not the raw file. The files.physionet.org mirror serves the
# zip directly.
CHALLENGE2015_URL = "https://physionet.org/files/challenge-2015/1.0.0/training.zip"
CHALLENGE2015_DIR = ROOT / "data" / "raw" / "challenge2015"

MIMIC_AF_URL = "https://zenodo.org/records/15906524/files/mimic_perform_af_csv.zip?download=1"
MIMIC_NONAF_URL = "https://zenodo.org/records/15906524/files/mimic_perform_non_af_csv.zip?download=1"
MIMIC_DIR = ROOT / "data" / "external" / "mimic_perform_af"


def _download_and_extract(url: str, dest_dir: Path, label: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] downloading {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    downloaded = 0
    chunk_size = 1 << 20
    for chunk in resp.iter_content(chunk_size=chunk_size):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"\r[{label}] {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
    print()
    buf.seek(0)
    print(f"[{label}] extracting to {dest_dir}")
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest_dir)
    print(f"[{label}] done")


def download_challenge2015() -> None:
    if any(CHALLENGE2015_DIR.rglob("*.hea")):
        print("[challenge2015] already present, skipping")
        return
    _download_and_extract(CHALLENGE2015_URL, CHALLENGE2015_DIR, "challenge2015")


def download_mimic_perform_af() -> None:
    if any(MIMIC_DIR.rglob("*.csv")):
        print("[mimic_perform_af] already present, skipping")
        return
    _download_and_extract(MIMIC_AF_URL, MIMIC_DIR / "af", "mimic_perform_af")
    _download_and_extract(MIMIC_NONAF_URL, MIMIC_DIR / "non_af", "mimic_perform_non_af")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["challenge2015", "mimic_perform_af", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.dataset in ("challenge2015", "all"):
        download_challenge2015()
    if args.dataset in ("mimic_perform_af", "all"):
        download_mimic_perform_af()


if __name__ == "__main__":
    main()
