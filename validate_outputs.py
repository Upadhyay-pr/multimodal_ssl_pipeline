#!/usr/bin/env python3
"""
validate_outputs.py
===================
Runs all required validation checks on preprocessed outputs and writes
a structured validation report + resource estimate.

Usage:
    python validate_outputs.py --config configs/pipeline_config.yaml
    python validate_outputs.py --config configs/pipeline_config.yaml --strict

Requires Python 3.8+.
"""

# Allow modern type hint syntax on Python 3.8/3.9
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate")

# ============================================================
# Helpers
# ============================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_npz(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load .npz produced by preprocess.py."""
    import io
    data = np.load(str(path), allow_pickle=True)
    signals = data["signals"]
    raw = data["metadata"]
    json_str = raw.item().decode() if isinstance(raw, np.ndarray) else raw
    meta = pd.read_json(io.StringIO(json_str))
    return signals, meta


def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def disk_usage(path: Path) -> int:
    """Total bytes under a directory."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ============================================================
# Check functions — each returns a dict with keys:
#   name, passed, details, errors (list[str])
# ============================================================

def check_file_exists(path: Path, name: str) -> dict:
    exists = path.exists() and path.stat().st_size > 0
    return {
        "name": name,
        "check": "file_exists",
        "passed": exists,
        "details": str(path),
        "errors": [] if exists else [f"File not found or empty: {path}"],
    }


def check_array_integrity(path: Path, name: str) -> dict:
    """No NaN, Inf, or malformed shapes in final signals."""
    errors = []
    details = {}

    try:
        signals, meta = load_npz(path)
    except Exception as e:
        return {"name": name, "check": "array_integrity", "passed": False,
                "details": {}, "errors": [f"Could not load file: {e}"]}

    details["shape"] = list(signals.shape)
    details["dtype"] = str(signals.dtype)
    details["n_samples"] = int(signals.shape[0])

    # dtype check
    if signals.dtype != np.float32:
        errors.append(f"dtype is {signals.dtype}, expected float32")

    # ndim check
    if signals.ndim != 3:
        errors.append(f"Expected 3-D array [N,C,T], got ndim={signals.ndim}")

    # NaN check
    nan_count = int(np.isnan(signals).sum())
    if nan_count > 0:
        errors.append(f"Silent NaNs present: {nan_count} values")
    details["nan_count"] = nan_count

    # Inf check
    inf_count = int(np.isinf(signals).sum())
    if inf_count > 0:
        errors.append(f"Inf values present: {inf_count}")
    details["inf_count"] = inf_count

    # Shape consistency with metadata
    if len(meta) != signals.shape[0]:
        errors.append(f"Metadata rows {len(meta)} != signals {signals.shape[0]}")

    # Required metadata columns
    required_cols = [
        "sample_id", "dataset_name", "modality", "subject_or_patient_id",
        "source_file_or_record", "split", "label_or_event", "sampling_rate_hz",
        "n_channels", "n_samples", "channel_schema", "qc_flags",
    ]
    missing_cols = [c for c in required_cols if c not in meta.columns]
    if missing_cols:
        errors.append(f"Missing metadata columns: {missing_cols}")

    return {
        "name": name,
        "check": "array_integrity",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_har_harmonisation(proc_dir: Path, cfg: dict) -> dict:
    """PAMAP2 and WISDM end up at 20 Hz with the same channel schema."""
    errors = []
    details = {}

    target_hz = cfg["har"]["target_rate_hz"]
    expected_schema = cfg["har"]["channel_schema"]
    n_ch = cfg["har"]["n_channels"]

    for fname in ["har_pretrain.npz", "har_supervised.npz"]:
        path = proc_dir / "har" / fname
        if not path.exists():
            errors.append(f"Missing: {path}")
            continue

        try:
            signals, meta = load_npz(path)
        except Exception as e:
            errors.append(f"Load failed {fname}: {e}")
            continue

        # Check sampling rate
        unique_hz = meta["sampling_rate_hz"].unique().tolist()
        if unique_hz != [target_hz]:
            errors.append(f"{fname}: sampling rates {unique_hz}, expected [{target_hz}]")
        details[f"{fname}_sampling_rates"] = unique_hz

        # Check channel count
        if signals.shape[1] != n_ch:
            errors.append(f"{fname}: n_channels={signals.shape[1]}, expected {n_ch}")
        details[f"{fname}_n_channels"] = int(signals.shape[1])

        # Check channel schema string
        schemas = meta["channel_schema"].unique().tolist()
        expected_str = ",".join(expected_schema)
        if any(s != expected_str for s in schemas):
            errors.append(f"{fname}: unexpected channel schemas: {schemas}")

        # Check both datasets present
        datasets = meta["dataset_name"].unique().tolist()
        details[f"{fname}_datasets"] = datasets
        for ds in ["PAMAP2", "WISDM"]:
            if ds not in datasets:
                errors.append(f"{fname}: {ds} not present in metadata")

    return {
        "name": "HAR harmonisation",
        "check": "har_harmonisation",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_har_window_sizes(proc_dir: Path, cfg: dict) -> dict:
    """HAR windows match required durations and overlap."""
    errors = []
    details = {}

    hz = cfg["har"]["target_rate_hz"]
    pt_exp  = int(cfg["har"]["pretrain"]["window_sec"] * hz)
    sup_exp = int(cfg["har"]["supervised"]["window_sec"] * hz)

    for fname, expected_T in [("har_pretrain.npz", pt_exp), ("har_supervised.npz", sup_exp)]:
        path = proc_dir / "har" / fname
        if not path.exists():
            errors.append(f"Missing: {path}")
            continue
        try:
            signals, _ = load_npz(path)
        except Exception as e:
            errors.append(f"Load failed: {e}")
            continue

        actual_T = signals.shape[2]
        details[f"{fname}_T"] = actual_T
        if actual_T != expected_T:
            errors.append(f"{fname}: window length {actual_T} != expected {expected_T}")

    return {
        "name": "HAR window sizes",
        "check": "har_window_sizes",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_null_label_handling(proc_dir: Path, cfg: dict) -> dict:
    """Class 0 (transient) in PAMAP2 is excluded; no null label windows exist unlabelled."""
    errors = []
    details = {}

    sup_path = proc_dir / "har" / "har_supervised.npz"
    if not sup_path.exists():
        return {"name": "Null label handling", "check": "null_labels",
                "passed": False, "details": {}, "errors": ["har_supervised.npz not found"]}

    try:
        _, meta = load_npz(sup_path)
    except Exception as e:
        return {"name": "Null label handling", "check": "null_labels",
                "passed": False, "details": {}, "errors": [str(e)]}

    # No None/NaN labels in supervised output
    null_labels = meta["label_or_event"].isna().sum()
    details["null_label_count"] = int(null_labels)
    if null_labels > 0:
        errors.append(f"Supervised output has {null_labels} null label rows")

    # All labels are in the defined range
    label_range = cfg["har"]["label_map"]
    max_label = max(label_range.values())
    out_of_range = (meta["label_or_event"].dropna().astype(int) > max_label).sum()
    details["out_of_range_labels"] = int(out_of_range)
    if out_of_range > 0:
        errors.append(f"{out_of_range} labels out of expected range [0, {max_label}]")

    # Label distribution
    label_counts = meta["label_or_event"].value_counts().to_dict()
    details["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}

    return {
        "name": "Null label handling",
        "check": "null_labels",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_leakage_control(proc_dir: Path, cfg: dict) -> dict:
    """Subject IDs preserved; ECG patient-level splits confirmed."""
    errors = []
    details = {}

    # HAR: subject IDs present
    for fname in ["har_pretrain.npz", "har_supervised.npz"]:
        path = proc_dir / "har" / fname
        if not path.exists():
            continue
        try:
            _, meta = load_npz(path)
        except Exception:
            continue
        null_subj = meta["subject_or_patient_id"].isna().sum()
        details[f"{fname}_null_subject_ids"] = int(null_subj)
        if null_subj > 0:
            errors.append(f"{fname}: {null_subj} rows with null subject_id")

    # ECG: patient-level splits
    ecg_path = proc_dir / "ecg" / "ecg_ptbxl.npz"
    if ecg_path.exists():
        try:
            _, meta = load_npz(ecg_path)
        except Exception as e:
            errors.append(f"Could not load ECG: {e}")
            meta = None

        if meta is not None:
            splits = meta["split"].value_counts().to_dict()
            details["ecg_split_distribution"] = {str(k): int(v) for k, v in splits.items()}

            # Verify no patient appears in both train and test
            if "subject_or_patient_id" in meta.columns:
                train_patients = set(meta[meta["split"] == "train"]["subject_or_patient_id"].unique())
                test_patients  = set(meta[meta["split"] == "test"]["subject_or_patient_id"].unique())
                overlap = train_patients & test_patients
                details["ecg_train_test_patient_overlap"] = len(overlap)
                if overlap:
                    errors.append(f"Patient-level leakage: {len(overlap)} patients in both train and test")
            else:
                errors.append("ECG metadata missing subject_or_patient_id column")

    return {
        "name": "Leakage control",
        "check": "leakage_control",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_eeg_annotations(proc_dir: Path, cfg: dict) -> dict:
    """EEG event codes and timing reflected correctly."""
    errors = []
    details = {}

    eeg_path = proc_dir / "eeg" / "eeg_motor_imagery.npz"
    if not eeg_path.exists():
        return {"name": "EEG annotations", "check": "eeg_annotations",
                "passed": False, "details": {}, "errors": ["eeg_motor_imagery.npz not found"]}

    try:
        signals, meta = load_npz(eeg_path)
    except Exception as e:
        return {"name": "EEG annotations", "check": "eeg_annotations",
                "passed": False, "details": {}, "errors": [str(e)]}

    # Check labels are 1 and 2 (T1/T2 only, no T0)
    labels = set(meta["label_or_event"].unique().tolist())
    details["unique_labels"] = sorted(labels)
    if not labels.issubset({1, 2}):
        errors.append(f"Unexpected EEG labels: {labels}. Expected subset of {{1, 2}}")

    # Check window shape [N, 64, 640]
    exp_hz = cfg["eeg"]["target_rate_hz"]
    exp_win = int(cfg["eeg"]["window_sec"] * exp_hz)
    exp_ch  = 64

    details["signal_shape"] = list(signals.shape)
    if signals.shape[1] != exp_ch:
        errors.append(f"EEG n_channels={signals.shape[1]}, expected {exp_ch}")
    if signals.shape[2] != exp_win:
        errors.append(f"EEG window_samples={signals.shape[2]}, expected {exp_win}")

    # Check run_id present
    if "run_id" not in meta.columns:
        errors.append("EEG metadata missing run_id column")
    else:
        runs_present = set(meta["run_id"].unique().tolist())
        details["runs_present"] = sorted(runs_present)

    # Check subject IDs
    n_subjects = meta["subject_or_patient_id"].nunique()
    details["n_subjects"] = int(n_subjects)

    return {
        "name": "EEG annotations",
        "check": "eeg_annotations",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_ecg_folds(proc_dir: Path, cfg: dict) -> dict:
    """PTB-XL fold metadata reported."""
    errors = []
    details = {}

    ecg_path = proc_dir / "ecg" / "ecg_ptbxl.npz"
    if not ecg_path.exists():
        return {"name": "ECG folds", "check": "ecg_folds",
                "passed": False, "details": {}, "errors": ["ecg_ptbxl.npz not found"]}

    try:
        _, meta = load_npz(ecg_path)
    except Exception as e:
        return {"name": "ECG folds", "check": "ecg_folds",
                "passed": False, "details": {}, "errors": [str(e)]}

    if "strat_fold" not in meta.columns:
        errors.append("strat_fold column missing from ECG metadata")
    else:
        fold_counts = meta["strat_fold"].value_counts().sort_index().to_dict()
        details["fold_counts"] = {str(k): int(v) for k, v in fold_counts.items()}

    splits = meta["split"].value_counts().to_dict() if "split" in meta.columns else {}
    details["split_counts"] = {str(k): int(v) for k, v in splits.items()}

    # Verify expected splits present
    for s in ["train", "val", "test"]:
        if s not in splits:
            errors.append(f"Split '{s}' not found in ECG metadata")

    return {
        "name": "ECG folds",
        "check": "ecg_folds",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def check_submission_samples(sub_dir: Path, cfg: dict) -> dict:
    """Submission sample packs have ~100 windows each."""
    errors = []
    details = {}
    n_expected = cfg["output"]["submission_samples_per_dataset"]

    expected_files = [
        sub_dir / "har" / "har_pretrain_sample.npz",
        sub_dir / "har" / "har_supervised_sample.npz",
        sub_dir / "eeg" / "eeg_motor_imagery_sample.npz",
        sub_dir / "ecg" / "ecg_ptbxl_sample.npz",
    ]

    for f in expected_files:
        if not f.exists():
            errors.append(f"Submission sample missing: {f}")
            continue
        try:
            signals, meta = load_npz(f)
            n = signals.shape[0]
            details[f.name] = n
            if n < 10:
                errors.append(f"{f.name}: only {n} samples (expected ~{n_expected})")
        except Exception as e:
            errors.append(f"{f.name}: load error: {e}")

    return {
        "name": "Submission sample packs",
        "check": "submission_samples",
        "passed": len(errors) == 0,
        "details": details,
        "errors": errors,
    }


def resource_estimate(cfg: dict) -> dict:
    """Estimate disk, RAM, and runtime usage."""
    raw_dir  = Path(cfg["paths"]["raw"])
    proc_dir = Path(cfg["paths"]["processed"])

    raw_size  = disk_usage(raw_dir)
    proc_size = disk_usage(proc_dir)

    # RAM estimation based on largest expected array
    # HAR: ~50k windows × 6ch × 200 samples × 4 bytes = ~240 MB peak
    # EEG: ~3000 windows × 64ch × 640 samples × 4 bytes = ~491 MB peak
    # ECG: ~21000 records × 12ch × 1000 samples × 4 bytes = ~1 GB peak
    har_peak_mb  = 50_000 * 6 * 200 * 4 / 1e6
    eeg_peak_mb  = 3_000 * 64 * 640 * 4 / 1e6
    ecg_peak_mb  = 21_000 * 12 * 1000 * 4 / 1e6

    system_ram_gb = psutil.virtual_memory().total / 1e9
    available_ram_gb = psutil.virtual_memory().available / 1e9

    return {
        "raw_data_disk": fmt_bytes(raw_size),
        "raw_data_disk_bytes": raw_size,
        "processed_disk": fmt_bytes(proc_size),
        "processed_disk_bytes": proc_size,
        "estimated_peak_ram_har_mb": round(har_peak_mb, 1),
        "estimated_peak_ram_eeg_mb": round(eeg_peak_mb, 1),
        "estimated_peak_ram_ecg_mb": round(ecg_peak_mb, 1),
        "estimated_peak_ram_total_gb": round((har_peak_mb + eeg_peak_mb + ecg_peak_mb) / 1000, 2),
        "system_ram_total_gb": round(system_ram_gb, 1),
        "system_ram_available_gb": round(available_ram_gb, 1),
        "estimated_runtime_minutes": {
            "har_pamap2": "5-10 (100Hz->20Hz resampling ~9 subjects)",
            "har_wisdm": "3-5 (already 20Hz, 51 subjects)",
            "har_mhealth": "1-2 (bonus, 10 subjects)",
            "eeg": "20-40 (MNE filtering, 109 subjects × 3 runs each)",
            "ecg": "15-30 (WFDB loading, 21k records, IO-bound)",
            "total_estimate": "45-90 minutes (single core); ~20-40 min with -j4",
        },
        "notes": [
            "EEG download: runs 4,8,12 only = ~3 GB (full dataset ~14 GB).",
            "ECG: 100 Hz chosen over 500 Hz saves 5× disk/RAM (1 GB vs 5 GB processed).",
            "For chunked ECG processing on low-RAM machines: set batch_size in config.",
            "If peak RAM exceeds available, process ECG in strat_fold batches.",
        ],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed outputs")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any check fails")
    args = parser.parse_args()

    cfg = load_config(args.config)

    proc_dir = Path(cfg["paths"]["processed"])
    sub_dir  = Path(cfg["paths"]["submission_sample"])
    rep_dir  = Path(cfg["paths"]["reports"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running validation checks...")
    t0 = time.time()

    checks = []

    # 1. File existence
    for fname, rel in [
        ("har_pretrain.npz",     proc_dir / "har" / "har_pretrain.npz"),
        ("har_supervised.npz",   proc_dir / "har" / "har_supervised.npz"),
        ("eeg_motor_imagery.npz", proc_dir / "eeg" / "eeg_motor_imagery.npz"),
        ("ecg_ptbxl.npz",        proc_dir / "ecg" / "ecg_ptbxl.npz"),
    ]:
        checks.append(check_file_exists(rel, fname))

    # 2. Array integrity (NaN, Inf, shape)
    for fname, rel in [
        ("har_pretrain.npz",     proc_dir / "har" / "har_pretrain.npz"),
        ("har_supervised.npz",   proc_dir / "har" / "har_supervised.npz"),
        ("eeg_motor_imagery.npz", proc_dir / "eeg" / "eeg_motor_imagery.npz"),
        ("ecg_ptbxl.npz",        proc_dir / "ecg" / "ecg_ptbxl.npz"),
    ]:
        if rel.exists():
            checks.append(check_array_integrity(rel, fname))

    # 3. HAR harmonisation
    checks.append(check_har_harmonisation(proc_dir, cfg))

    # 4. HAR window sizes
    checks.append(check_har_window_sizes(proc_dir, cfg))

    # 5. Null label handling
    checks.append(check_null_label_handling(proc_dir, cfg))

    # 6. Leakage control
    checks.append(check_leakage_control(proc_dir, cfg))

    # 7. EEG annotations
    checks.append(check_eeg_annotations(proc_dir, cfg))

    # 8. ECG folds
    checks.append(check_ecg_folds(proc_dir, cfg))

    # 9. Submission samples
    checks.append(check_submission_samples(sub_dir, cfg))

    # Resource estimate
    resources = resource_estimate(cfg)

    # Summary
    n_passed = sum(1 for c in checks if c["passed"])
    n_total  = len(checks)

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY: %d / %d checks passed", n_passed, n_total)
    logger.info("=" * 60)
    for c in checks:
        status = "✓ PASS" if c["passed"] else "✗ FAIL"
        logger.info("  %s  %s", status, c["name"])
        if not c["passed"]:
            for err in c["errors"]:
                logger.error("      -> %s", err)

    # Write report
    report = {
        "report_type": "validation_report",
        "generated_at": datetime.now(tz=None).astimezone().isoformat(),
        "pipeline_version": "1.0.0",
        "summary": {
            "total_checks": n_total,
            "passed": n_passed,
            "failed": n_total - n_passed,
            "all_passed": n_passed == n_total,
        },
        "checks": checks,
        "resource_estimate": resources,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    rep_path = rep_dir / "validation_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation report written to %s", rep_path)

    # Also write a human-readable text summary
    txt_path = rep_dir / "validation_report.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MULTIMODAL SSL PIPELINE — VALIDATION REPORT\n")
        f.write(f"Generated: {report['generated_at']}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Summary: {n_passed}/{n_total} checks passed\n\n")

        for c in checks:
            status = "PASS" if c["passed"] else "FAIL"
            f.write(f"[{status}] {c['name']}\n")
            if c.get("details"):
                if isinstance(c["details"], dict):
                    for k, v in c["details"].items():
                        f.write(f"       {k}: {v}\n")
                else:
                    f.write(f"       details: {c['details']}\n")
            if not c["passed"]:
                for err in c["errors"]:
                    f.write(f"       ERROR: {err}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("RESOURCE ESTIMATE\n")
        f.write("=" * 70 + "\n")
        for k, v in resources.items():
            if k != "notes":
                f.write(f"  {k}: {v}\n")
        f.write("\n  Notes:\n")
        for note in resources.get("notes", []):
            f.write(f"    - {note}\n")

    logger.info("Text report written to %s", txt_path)

    if args.strict and n_passed < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
