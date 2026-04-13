#!/usr/bin/env python3
"""
download_datasets.py
====================
Pure-Python fallback downloader for all pipeline datasets.
Uses only Python standard library (urllib) — no extra dependencies needed.

This script is called automatically by setup_data.sh when wget is unavailable
(e.g. macOS without homebrew wget, or Windows without wget in PATH).
You can also run it directly for individual datasets.

Usage:
    python download_datasets.py                        # download all datasets
    python download_datasets.py --dataset pamap2       # single dataset
    python download_datasets.py --dataset eegmmidb     # EEG only
    python download_datasets.py --dataset ptbxl        # ECG only
    python download_datasets.py --dry-run              # print only, no download
    python download_datasets.py --list                 # list available datasets

How it works:
    - ZIP files (PAMAP2, WISDM, mHealth): direct HTTP download
    - PhysioNet files (EEGMMIDB, PTB-XL): HTTP with redirect following
    - Retries up to 3 times on failure
    - Skips files already downloaded (checks file size)
    - Prints clear progress and error messages

Requires Python 3.8+.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

# ============================================================
# Dataset registry
# ============================================================

DATASETS = {
    "pamap2": {
        "name": "PAMAP2 Physical Activity Monitoring",
        "type": "zip",
        "urls": [
            "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        ],
        "dest_dir": "data/raw/pamap2",
        "filename": "PAMAP2_Dataset.zip",
        "mandatory": True,
        "note": "100 Hz wrist+chest+ankle IMU; 9 subjects; 18 activities",
    },
    "wisdm": {
        "name": "WISDM Smartphone and Smartwatch Activity",
        "type": "zip",
        "urls": [
            "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip",
        ],
        "dest_dir": "data/raw/wisdm",
        "filename": "wisdm-dataset.zip",
        "mandatory": True,
        "note": "20 Hz phone+watch acc+gyr; 51 subjects; 18 activities",
    },
    "mhealth": {
        "name": "MHEALTH Mobile Health Dataset (bonus)",
        "type": "zip",
        "urls": [
            "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
        ],
        "dest_dir": "data/raw/mhealth",
        "filename": "MHEALTHDATASET.zip",
        "mandatory": False,
        "note": "Bonus: 50 Hz; chest+wrist+ankle; 2-lead ECG; 10 subjects",
    },
    "eegmmidb": {
        "name": "EEG Motor Movement/Imagery Dataset",
        "type": "physionet_eeg",
        "base_url": "https://physionet.org/files/eegmmidb/1.0.0",
        "dest_dir": "data/raw/eegmmidb",
        "mandatory": True,
        "note": "160 Hz EDF+; 64-ch; 109 subjects; runs 4,8,12 only (~3 GB)",
    },
    "ptbxl": {
        "name": "PTB-XL ECG Dataset",
        "type": "physionet_ptbxl",
        "base_url": "https://physionet.org/files/ptb-xl/1.0.3",
        "dest_dir": "data/raw/ptbxl",
        "mandatory": True,
        "note": "12-lead ECG; 21801 records; 18869 patients; 100/500 Hz (~1.7 GB)",
    },
}


# ============================================================
# Download utilities
# ============================================================

def _progress_bar(downloaded: int, total: int, bar_len: int = 40) -> str:
    """Return a simple ASCII progress bar string."""
    if total <= 0:
        return f"  {downloaded // 1024 // 1024} MB downloaded"
    frac = downloaded / total
    filled = int(bar_len * frac)
    bar = "=" * filled + "-" * (bar_len - filled)
    pct = frac * 100
    mb_done = downloaded / 1024 / 1024
    mb_total = total / 1024 / 1024
    return f"  [{bar}] {pct:.1f}%  {mb_done:.1f}/{mb_total:.1f} MB"


def download_url(url: str, dest_path: Path, description: str = "", retries: int = 3,
                 dry_run: bool = False) -> bool:
    """
    Download a single URL to dest_path.

    Returns True on success, False on failure.
    Skips if the file already exists and is larger than 10 KB.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would download: {url}")
        print(f"            → {dest_path}")
        return True

    # Skip if already present and non-trivial
    if dest_path.exists() and dest_path.stat().st_size > 10_000:
        print(f"  Already downloaded ({dest_path.stat().st_size // 1024} KB): {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt}/{retries}: {description or url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 multimodal-ssl-pipeline/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536  # 64 KB chunks

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Show progress on same line
                        print(f"\r{_progress_bar(downloaded, total_size)}", end="", flush=True)

            print()  # newline after progress bar

            # Verify the file is non-trivial
            if tmp_path.exists() and tmp_path.stat().st_size > 1000:
                tmp_path.rename(dest_path)
                print(f"  OK: {dest_path.name} ({dest_path.stat().st_size // 1024} KB)")
                return True
            else:
                print(f"  File too small ({tmp_path.stat().st_size} bytes). Retrying...")
                tmp_path.unlink(missing_ok=True)

        except urllib.error.HTTPError as e:
            print(f"\n  HTTP {e.code} from {url}")
        except urllib.error.URLError as e:
            print(f"\n  Network error: {e.reason}")
        except Exception as e:
            print(f"\n  Unexpected error: {e}")

        if attempt < retries:
            wait = attempt * 3
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)

    tmp_path.unlink(missing_ok=True)
    return False


def extract_zip(zip_path: Path, dest_dir: Path, dataset_name: str) -> None:
    """Extract a ZIP archive if not already extracted."""
    existing = [f for f in dest_dir.rglob("*") if f.is_file() and f.suffix != ".zip"]
    if existing:
        print(f"  Already extracted ({len(existing)} files present).")
        return

    print(f"  Extracting {zip_path.name} → {dest_dir} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for i, member in enumerate(members, 1):
                zf.extract(member, dest_dir)
                if i % 500 == 0 or i == len(members):
                    print(f"\r  Extracted {i}/{len(members)} files...", end="", flush=True)
        print()
        print(f"  Done: {len(members)} files extracted.")
    except zipfile.BadZipFile:
        print(f"  ERROR: {zip_path.name} is not a valid ZIP file. Delete it and re-download.")
    except Exception as e:
        print(f"  ERROR extracting {zip_path.name}: {e}")


# ============================================================
# Kaggle mirror URLs for when UCI ML Repository is down
# ============================================================

KAGGLE_MIRRORS = {
    "PAMAP2": {
        "url": "https://www.kaggle.com/datasets/nandinibagga/pamap2-dataset",
        "alt_url": "https://www.kaggle.com/datasets/jeetp22/pamap2",
        "dest_hint": "data/raw/pamap2/",
        "note": "IMPORTANT: Download must contain Protocol/subject101.dat ... subject109.dat (raw .dat files).\n"
                "           If the download only has .npy files, try the alt_url above or search Kaggle for 'PAMAP2 Protocol'.",
    },
    "WISDM": {
        "url": "https://www.kaggle.com/datasets/mashlyn/smartphone-and-smartwatch-activity-and-biometrics",
        "dest_hint": "data/raw/wisdm/wisdm-dataset.zip",
        "note": "Should contain wisdm-dataset/raw/watch/accel/*.txt and wisdm-dataset/raw/watch/gyro/*.txt",
    },
    "mHealth": {
        "url": "https://www.kaggle.com/datasets/gaurav2022/mobile-health",
        "alt_url": "https://www.kaggle.com/datasets/nirmalsankalana/mhealth-dataset-data-set",
        "dest_hint": "data/raw/mhealth/MHEALTHDATASET.zip",
        "note": "The pipeline can handle both the per-subject .log files (original) and single CSV (Kaggle reformatted).",
    },
}


def _print_download_help(dataset_name: str, dest_path: Path, mandatory: bool) -> None:
    """Print helpful troubleshooting when a UCI download fails."""
    print()
    print("=" * 60)
    print(f"  DOWNLOAD FAILED: {dataset_name}")
    print("=" * 60)
    print()
    print("  The UCI ML Repository (archive.ics.uci.edu) may be")
    print("  temporarily unavailable (502/503 errors are server-side).")
    print()

    # Check which mirror we have
    mirror = KAGGLE_MIRRORS.get(dataset_name)
    if mirror:
        print("  OPTION 1 — Download manually from Kaggle:")
        print(f"    Primary:   {mirror['url']}")
        if mirror.get("alt_url"):
            print(f"    Alternate: {mirror['alt_url']}")
        print(f"    1. Open in browser, click 'Download' (free Kaggle account required)")
        print(f"    2. Move the file to: {mirror['dest_hint']}")
        print(f"    3. Extract: unzip <file>.zip -d {dest_path.parent}/")
        if mirror.get("note"):
            print(f"    NOTE: {mirror['note']}")
        print()

    print("  OPTION 2 — Wait for UCI to come back online, then re-run:")
    print("    python download_datasets.py")
    print("    (Already-downloaded files are skipped automatically.)")
    print()
    print("  OPTION 3 — Direct download with curl or wget:")
    print(f"    curl -L -o '{dest_path}' '<UCI_URL>'")
    print(f"    wget -O '{dest_path}' '<UCI_URL>'")
    print()

    if mandatory:
        print(f"  NOTE: {dataset_name} is MANDATORY for the pipeline.")
    else:
        print(f"  NOTE: {dataset_name} is OPTIONAL (bonus). Continuing without it.")
    print("=" * 60)
    print()


# ============================================================
# Dataset-specific downloaders
# ============================================================

def download_zip_dataset(cfg: dict, dry_run: bool = False) -> bool:
    """Download and extract a UCI ZIP dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg['name']}")
    print(f"{'='*60}")

    dest_dir = Path(cfg["dest_dir"])
    dest_path = dest_dir / cfg["filename"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    success = False
    for url in cfg["urls"]:
        if download_url(url, dest_path, description=cfg["name"], dry_run=dry_run):
            success = True
            break
        print(f"  URL failed: {url}  (trying next...)")

    if not success:
        _print_download_help(cfg["name"], dest_path, cfg.get("mandatory", False))
        return False

    if not dry_run and dest_path.exists():
        extract_zip(dest_path, dest_dir, cfg["name"])

    return True


def download_eegmmidb(cfg: dict, dry_run: bool = False) -> bool:
    """
    Download EEGMMIDB motor imagery runs 4, 8, 12 for all 109 subjects.
    Downloads individual EDF files (one per subject per run).
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg['name']}")
    print(f"Runs: 4, 8, 12 only  (motor imagery: left/right fist)")
    print(f"{'='*60}")

    base_url = cfg["base_url"]
    dest_dir = Path(cfg["dest_dir"])

    # Count existing EDF files
    existing_edfs = list(dest_dir.rglob("*.edf")) if dest_dir.exists() else []
    if len(existing_edfs) > 200:
        print(f"  {len(existing_edfs)} EDF files already present. Skipping.")
        return True

    print(f"  Downloading 109 subjects × 3 runs = 327 EDF files...")
    print(f"  (existing: {len(existing_edfs)})")

    if dry_run:
        print(f"  [DRY-RUN] Would download to {dest_dir}/S001/S001R04.edf ... S109/S109R12.edf")
        return True

    downloaded = 0
    skipped = 0
    failed = 0

    for subj_num in range(1, 110):
        subj_id = f"S{subj_num:03d}"
        subj_dir = dest_dir / subj_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        for run_num in [4, 8, 12]:
            run_str = f"{run_num:02d}"
            filename = f"{subj_id}R{run_str}.edf"
            out_path = subj_dir / filename
            url = f"{base_url}/{subj_id}/{filename}"

            if out_path.exists() and out_path.stat().st_size > 1000:
                skipped += 1
                continue

            ok = download_url(url, out_path, description=f"{subj_id} run {run_num}",
                               retries=2, dry_run=False)
            if ok:
                downloaded += 1
            else:
                failed += 1
                # Remove partial file
                out_path.unlink(missing_ok=True)

        # Progress every 10 subjects
        if subj_num % 10 == 0 or subj_num == 109:
            total_done = downloaded + skipped
            print(f"  Progress: {subj_num}/109 subjects  "
                  f"(downloaded={downloaded}, skipped={skipped}, failed={failed})")

    total_edf = len(list(dest_dir.rglob("*.edf")))
    print(f"\n  Done: {total_edf} EDF files total (failed: {failed})")

    if total_edf < 10 and cfg.get("mandatory"):
        print(f"\n  ERROR: Too few EDF files ({total_edf}). Check PhysioNet connectivity.")
        print(f"  Re-run to retry missing files (script is idempotent).")
        return False

    return True


def download_ptbxl(cfg: dict, dry_run: bool = False) -> bool:
    """
    Download PTB-XL from PhysioNet.
    Downloads the database index files and records at 100 Hz.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg['name']}")
    print(f"{'='*60}")

    base_url = cfg["base_url"]
    dest_dir = Path(cfg["dest_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [DRY-RUN] Would download PTB-XL from {base_url}")
        return True

    # Check if already downloaded
    existing = list(dest_dir.rglob("*.hea")) if dest_dir.exists() else []
    if len(existing) > 1000:
        print(f"  {len(existing)} .hea files already present. Skipping.")
        return True

    # Download the main metadata files first
    print("  Downloading PTB-XL metadata files...")
    metadata_files = [
        "ptbxl_database.csv",
        "scp_statements.csv",
        "RECORDS",
        "ANNOTATORS",
        "REVISIONS",
    ]
    for fname in metadata_files:
        url = f"{base_url}/{fname}"
        dest_path = dest_dir / fname
        download_url(url, dest_path, description=f"PTB-XL: {fname}", retries=2, dry_run=False)

    # Read the RECORDS file to get the list of all record paths
    records_file = dest_dir / "RECORDS"
    if not records_file.exists():
        print("  ERROR: Could not download RECORDS index file.")
        print("  This file lists all 21,801 ECG records to download.")
        if cfg.get("mandatory"):
            print("  PTB-XL download failed. Manual alternative:")
            print(f"    cd {dest_dir}")
            print(f"    wget -r -np -nc -q {base_url}/")
        return False

    records = records_file.read_text().strip().splitlines()
    print(f"  Found {len(records)} records in RECORDS index.")
    print(f"  Downloading .dat and .hea files for all records...")
    print(f"  (existing .hea: {len(existing)})")

    downloaded = 0
    failed = 0

    for i, record_path in enumerate(records, 1):
        record_path = record_path.strip()
        if not record_path:
            continue

        # record_path is like "records100/00001/00001_lr"
        # We need the .dat and .hea files
        for ext in [".hea", ".dat"]:
            url = f"{base_url}/{record_path}{ext}"
            out_path = dest_dir / f"{record_path}{ext}"

            if out_path.exists() and out_path.stat().st_size > 100:
                continue  # already downloaded

            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = download_url(url, out_path, description="",
                               retries=2, dry_run=False)
            if ok:
                downloaded += 1
            else:
                failed += 1

        if i % 1000 == 0 or i == len(records):
            print(f"  Progress: {i}/{len(records)} records  "
                  f"(downloaded={downloaded}, failed={failed})")

    print(f"\n  Done: {downloaded} files downloaded ({failed} failed).")

    # Final check
    hea_count = len(list(dest_dir.rglob("*.hea")))
    if hea_count < 100 and cfg.get("mandatory"):
        print(f"  WARNING: Only {hea_count} .hea files found (expected ~21801).")
        print("  Re-run to download missing files.")

    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pure-Python dataset downloader for the multimodal SSL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        help="Override destination directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    parser.add_argument(
        "--no-mhealth",
        action="store_true",
        help="Skip optional mHealth bonus dataset",
    )
    parser.add_argument(
        "--skip-eeg",
        action="store_true",
        help="Skip EEGMMIDB (~3 GB)",
    )
    parser.add_argument(
        "--skip-ecg",
        action="store_true",
        help="Skip PTB-XL (~1.7 GB)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for key, cfg in DATASETS.items():
            mandatory_str = "mandatory" if cfg.get("mandatory") else "optional (bonus)"
            print(f"  {key:12s}  [{mandatory_str}]  {cfg['name']}")
            print(f"              {cfg['note']}")
            print()
        return

    # Ensure we're in the project root (where setup_data.sh is)
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print(f"\nMultimodal SSL Pipeline — Dataset Downloader")
    print(f"Working directory: {script_dir}")
    if args.dry_run:
        print("DRY-RUN mode: no files will be downloaded")

    # Determine which datasets to download
    if args.dataset == "all":
        targets = list(DATASETS.keys())
    else:
        targets = [args.dataset]

    if args.no_mhealth and "mhealth" in targets:
        targets.remove("mhealth")
    if args.skip_eeg and "eegmmidb" in targets:
        targets.remove("eegmmidb")
    if args.skip_ecg and "ptbxl" in targets:
        targets.remove("ptbxl")

    results = {}
    for key in targets:
        cfg = DATASETS[key].copy()
        if args.dest_dir:
            cfg["dest_dir"] = args.dest_dir

        if cfg["type"] == "zip":
            ok = download_zip_dataset(cfg, dry_run=args.dry_run)
        elif cfg["type"] == "physionet_eeg":
            ok = download_eegmmidb(cfg, dry_run=args.dry_run)
        elif cfg["type"] == "physionet_ptbxl":
            ok = download_ptbxl(cfg, dry_run=args.dry_run)
        else:
            print(f"Unknown dataset type: {cfg['type']}")
            ok = False

        results[key] = ok

    # Summary
    print(f"\n{'='*60}")
    print("Download summary:")
    all_mandatory_ok = True
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        mandatory = "mandatory" if DATASETS[key].get("mandatory") else "optional"
        print(f"  {key:12s} [{mandatory:9s}]  {status}")
        if not ok and DATASETS[key].get("mandatory"):
            all_mandatory_ok = False

    if not all_mandatory_ok:
        print("\nOne or more mandatory datasets failed to download.")
        print("Re-run this script to retry (downloads are idempotent).")
        sys.exit(1)
    else:
        print("\nAll mandatory datasets downloaded successfully.")
        print("\nNext step:")
        print("  python preprocess.py --config configs/pipeline_config.yaml")


if __name__ == "__main__":
    main()
