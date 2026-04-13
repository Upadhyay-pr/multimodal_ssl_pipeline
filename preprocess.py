#!/usr/bin/env python3
"""
preprocess.py
=============
Main preprocessing script for the multimodal SSL pipeline.
Handles HAR (PAMAP2, WISDM, mHealth), EEG (EEGMMIDB), and ECG (PTB-XL).

Usage:
    python preprocess.py --config configs/pipeline_config.yaml
    python preprocess.py --config configs/pipeline_config.yaml --modality har
    python preprocess.py --config configs/pipeline_config.yaml --modality eeg
    python preprocess.py --config configs/pipeline_config.yaml --modality ecg

Requires Python 3.8+.
"""

# Allow modern type hint syntax (e.g. str | Path, list[dict]) on Python 3.8/3.9
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# ============================================================
# Utility
# ============================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npz(out_path: Path, signals: np.ndarray, metadata: list[dict]) -> None:
    """Save signals [N, C, T] float32 and metadata list to .npz."""
    signals = signals.astype(np.float32)
    meta_df = pd.DataFrame(metadata)
    np.savez_compressed(
        str(out_path),
        signals=signals,
        metadata=meta_df.to_json(orient="records").encode(),
    )
    logger.info("  Saved %s  shape=%s  dtype=%s", out_path.name, signals.shape, signals.dtype)


def resample_signal(sig: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    """Resample [C, T] signal from orig_hz to target_hz using polyphase resampling."""
    if orig_hz == target_hz:
        return sig
    from math import gcd
    g = gcd(int(target_hz), int(orig_hz))
    up = int(target_hz) // g
    down = int(orig_hz) // g
    return sp_signal.resample_poly(sig, up, down, axis=-1)


def bandpass_filter(sig: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass to [C, T] signal."""
    nyq = fs / 2.0
    lo = max(lowcut / nyq, 0.001)
    hi = min(highcut / nyq, 0.999)
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    return sp_signal.filtfilt(b, a, sig, axis=-1)


def notch_filter(sig: np.ndarray, freq: float, fs: int, quality: float = 30.0) -> np.ndarray:
    """Apply notch filter at freq Hz."""
    b, a = sp_signal.iirnotch(freq, quality, fs)
    return sp_signal.filtfilt(b, a, sig, axis=-1)


def zscore_normalize(sig: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation. [C, T]."""
    mean = sig.mean(axis=-1, keepdims=True)
    std = sig.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (sig - mean) / std


def interpolate_missing(arr: np.ndarray, max_gap: int = 10) -> tuple[np.ndarray, bool]:
    """
    Interpolate NaN runs of length <= max_gap in a 1-D array.
    Returns (filled_array, was_valid).
    was_valid=False if any NaN run > max_gap exists.
    """
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr, True

    indices = np.arange(len(arr))
    valid = ~nan_mask

    # Check max gap
    in_gap = False
    gap_len = 0
    for v in nan_mask:
        if v:
            gap_len += 1
            in_gap = True
        else:
            if in_gap and gap_len > max_gap:
                return arr, False
            gap_len = 0
            in_gap = False
    if in_gap and gap_len > max_gap:
        return arr, False

    # Linear interpolate
    f = interp1d(indices[valid], arr[valid], bounds_error=False, fill_value="extrapolate")
    arr_filled = arr.copy()
    arr_filled[nan_mask] = f(indices[nan_mask])
    return arr_filled, True


def window_signal(sig: np.ndarray, window_samples: int, step_samples: int) -> np.ndarray:
    """
    Slide a window over [C, T] signal.
    Returns [N_windows, C, window_samples].
    """
    C, T = sig.shape
    starts = range(0, T - window_samples + 1, step_samples)
    windows = []
    for s in starts:
        windows.append(sig[:, s:s + window_samples])
    if not windows:
        return np.empty((0, C, window_samples), dtype=np.float32)
    return np.stack(windows, axis=0)


def make_sample_id(prefix: str, subject: str, record: str, window_idx: int) -> str:
    return f"{prefix}_sub{subject}_rec{record}_w{window_idx:05d}"


# ============================================================
# HAR: PAMAP2
# ============================================================

# PAMAP2 column indices (0-indexed, after splitting by whitespace)
# Col 0: timestamp, Col 1: activity_id
# Wrist IMU (hand): cols 4-6 acc 16g, cols 7-9 acc 6g, cols 10-12 gyro, cols 13-15 mag
# We use: wrist acc 16g = cols 4,5,6  | wrist gyro = cols 10,11,12
PAMAP2_ACC_COLS = [4, 5, 6]    # hand accelerometer 16g x,y,z  (100 Hz)
PAMAP2_GYR_COLS = [10, 11, 12] # hand gyroscope x,y,z
PAMAP2_ACTIVITY_COL = 1
PAMAP2_N_COLS = 54             # total columns per row

def load_pamap2(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Load all PAMAP2 .dat files from Protocol/ subfolder.
    Returns a DataFrame with columns: subject_id, timestamp, activity_id,
    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z.
    """
    protocol_dir = None
    for candidate in [
        raw_dir / "PAMAP2_Dataset" / "Protocol",
        raw_dir / "Protocol",
    ]:
        if candidate.exists():
            protocol_dir = candidate
            break

    if protocol_dir is None:
        # Detect if user downloaded the Kaggle preprocessed version instead of raw UCI data
        npy_files = list(raw_dir.glob("*.npy"))
        if npy_files:
            raise FileNotFoundError(
                f"PAMAP2 Protocol dir not found under {raw_dir}.\n"
                f"  Found .npy files instead: {[f.name for f in npy_files]}\n"
                f"  These are PREPROCESSED files from a Kaggle mirror, not the original UCI data.\n"
                f"  The pipeline needs the RAW .dat files to perform its own channel selection,\n"
                f"  resampling, and label handling as required by the assessment.\n\n"
                f"  HOW TO FIX:\n"
                f"  1. Delete the current files: rm data/raw/pamap2/*.npy data/raw/pamap2/*.zip\n"
                f"  2. Download the ORIGINAL PAMAP2 from one of these Kaggle mirrors:\n"
                f"     - https://www.kaggle.com/datasets/nandinibagga/pamap2-dataset\n"
                f"     - https://www.kaggle.com/datasets/jeetp22/pamap2\n"
                f"     Search for one containing: PAMAP2_Dataset/Protocol/subject101.dat\n"
                f"  3. Move the zip to data/raw/pamap2/ and extract:\n"
                f"     unzip <downloaded_file>.zip -d data/raw/pamap2/\n"
                f"  4. Re-run: python preprocess.py --modality har\n\n"
                f"  If UCI is back online, just re-run: bash setup_data.sh"
            )
        raise FileNotFoundError(
            f"PAMAP2 Protocol dir not found under {raw_dir}.\n"
            f"  Expected: {raw_dir}/PAMAP2_Dataset/Protocol/subject101.dat\n"
            f"  Ensure PAMAP2_Dataset.zip has been downloaded and extracted.\n"
            f"  Run: bash setup_data.sh  (or see README troubleshooting section)"
        )

    logger.info("  PAMAP2 loading from %s", protocol_dir)
    dfs = []
    for dat_file in sorted(protocol_dir.glob("subject*.dat")):
        subject_id = int(dat_file.stem.replace("subject", ""))
        try:
            df = pd.read_csv(dat_file, sep=r"\s+", header=None, dtype=float, na_values=["NaN"])
        except Exception as e:
            logger.warning("    Could not parse %s: %s", dat_file.name, e)
            continue

        if df.shape[1] < 13:
            logger.warning("    Unexpected column count %d in %s", df.shape[1], dat_file.name)
            continue

        out = pd.DataFrame({
            "subject_id": subject_id,
            "timestamp":  df.iloc[:, 0].values,
            "activity_id": df.iloc[:, PAMAP2_ACTIVITY_COL].astype(int, errors="ignore"),
            "acc_x": df.iloc[:, PAMAP2_ACC_COLS[0]].values,
            "acc_y": df.iloc[:, PAMAP2_ACC_COLS[1]].values,
            "acc_z": df.iloc[:, PAMAP2_ACC_COLS[2]].values,
            "gyr_x": df.iloc[:, PAMAP2_GYR_COLS[0]].values,
            "gyr_y": df.iloc[:, PAMAP2_GYR_COLS[1]].values,
            "gyr_z": df.iloc[:, PAMAP2_GYR_COLS[2]].values,
            "source_file": dat_file.name,
        })
        dfs.append(out)
        logger.info("    Loaded %s: %d rows", dat_file.name, len(out))

    if not dfs:
        raise RuntimeError("No PAMAP2 .dat files loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("  PAMAP2 total rows: %d", len(combined))
    return combined


def preprocess_pamap2(raw_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    Returns:
        pretrain_signals  [N, 6, 200]  (10s @ 20Hz, no label)
        supervised_signals [M, 6, 100] (5s @ 20Hz, 50% overlap, labelled)
        pretrain_meta, supervised_meta
    """
    har_cfg = cfg["har"]
    source_hz = cfg["datasets"]["pamap2"]["source_rate_hz"]  # 100
    target_hz = har_cfg["target_rate_hz"]                    # 20

    pretrain_ws  = int(har_cfg["pretrain"]["window_sec"] * target_hz)   # 200
    supervised_ws = int(har_cfg["supervised"]["window_sec"] * target_hz) # 100
    supervised_step = int(supervised_ws * (1 - har_cfg["supervised"]["overlap_frac"]))

    label_map = har_cfg["pamap2_label_map"]         # int -> str
    unified_map = har_cfg["label_map"]               # str -> int

    df = load_pamap2(raw_dir, cfg)

    # Drop transient (activity=0) and activities not in our unified schema
    df["activity_id"] = df["activity_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df[df["activity_id"].astype(str) != "0"]  # drop null/transient
    df = df[df["activity_id"].astype(str).isin([str(k) for k in label_map])]

    # Map to unified label string
    df["label_str"] = df["activity_id"].astype(int).map(label_map)
    df["label_int"] = df["label_str"].map(unified_map)

    signal_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

    max_gap = har_cfg["max_interpolation_gap_samples"]
    pretrain_signals, pretrain_meta = [], []
    sup_signals, sup_meta = [], []

    for subj in sorted(df["subject_id"].unique()):
        subj_df = df[df["subject_id"] == subj].copy()

        # Interpolate missing values per channel, per contiguous activity segment
        for col in signal_cols:
            arr = subj_df[col].values.astype(float)
            filled, ok = interpolate_missing(arr, max_gap)
            if not ok:
                logger.debug("    PAMAP2 subj %s col %s: large gap found, using ffill", subj, col)
                series = pd.Series(arr).interpolate(method="linear", limit=max_gap)
                series = series.ffill().bfill()
                filled = series.values
            subj_df[col] = filled

        # Resample 100 Hz -> 20 Hz
        sig = subj_df[signal_cols].values.T.astype(float)  # [6, T]
        sig = resample_signal(sig, source_hz, target_hz)    # [6, T']

        # Resample labels (take mode in each resample block)
        n_orig = len(subj_df)
        n_new  = sig.shape[1]
        ratio  = n_orig / n_new
        label_arr = subj_df["label_int"].values
        source_file = subj_df["source_file"].iloc[0]

        def get_label_at(idx):
            orig_idx = min(int(idx * ratio), n_orig - 1)
            return label_arr[orig_idx]

        # --- PRETRAIN windows (no overlap, no labels) ---
        pt_wins = window_signal(sig, pretrain_ws, pretrain_ws)
        for w_idx, w in enumerate(pt_wins):
            pretrain_signals.append(w)
            pretrain_meta.append({
                "sample_id": make_sample_id("pamap2_pretrain", str(subj), "protocol", w_idx),
                "dataset_name": "PAMAP2",
                "modality": "HAR",
                "subject_or_patient_id": str(subj),
                "source_file_or_record": source_file,
                "split": "pretrain",
                "label_or_event": None,
                "sampling_rate_hz": target_hz,
                "n_channels": w.shape[0],
                "n_samples": w.shape[1],
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "",
            })

        # --- SUPERVISED windows (50% overlap, labelled) ---
        sup_wins = window_signal(sig, supervised_ws, supervised_step)
        for w_idx, w in enumerate(sup_wins):
            # Majority label in window
            start_sample = w_idx * supervised_step
            end_sample   = start_sample + supervised_ws
            win_labels = [get_label_at(i) for i in range(start_sample, min(end_sample, n_new))]
            win_labels_clean = [l for l in win_labels if l is not None and not np.isnan(l)]
            if not win_labels_clean:
                continue
            majority_label = max(set(win_labels_clean), key=win_labels_clean.count)

            qc = []
            if np.any(np.isnan(w)):
                qc.append("nan_present")
            if np.any(np.isinf(w)):
                qc.append("inf_present")

            sup_signals.append(w)
            sup_meta.append({
                "sample_id": make_sample_id("pamap2_sup", str(subj), "protocol", w_idx),
                "dataset_name": "PAMAP2",
                "modality": "HAR",
                "subject_or_patient_id": str(subj),
                "source_file_or_record": source_file,
                "split": "supervised",
                "label_or_event": int(majority_label),
                "sampling_rate_hz": target_hz,
                "n_channels": w.shape[0],
                "n_samples": w.shape[1],
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "|".join(qc),
            })

    pt_arr  = np.stack(pretrain_signals,  axis=0).astype(np.float32) if pretrain_signals else np.empty((0, 6, pretrain_ws), np.float32)
    sup_arr = np.stack(sup_signals, axis=0).astype(np.float32) if sup_signals else np.empty((0, 6, supervised_ws), np.float32)
    logger.info("  PAMAP2 pretrain windows: %d  supervised windows: %d", len(pt_arr), len(sup_arr))
    return pt_arr, sup_arr, pretrain_meta, sup_meta


# ============================================================
# HAR: WISDM
# ============================================================

def load_wisdm(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Load WISDM watch accelerometer + watch gyroscope.
    Returns DataFrame with: subject_id, activity_code, acc_x,y,z, gyr_x,y,z, source_file.
    """
    # Try to find the watch subdirectories
    watch_acc_dir = None
    watch_gyr_dir = None

    # Search multiple possible nesting levels (zip extraction can create double-nested dirs)
    search_roots = [
        raw_dir / "wisdm-dataset",
        raw_dir,
        raw_dir / "wisdm-dataset" / "wisdm-dataset",   # double-nested from zip
    ]
    # Also dynamically find any subdirectory that contains raw/watch/accel
    for child in raw_dir.rglob("raw"):
        if (child / "watch" / "accel").exists():
            search_roots.insert(0, child.parent)

    for candidate_root in search_roots:
        acc_c = candidate_root / "raw" / "watch" / "accel"
        gyr_c = candidate_root / "raw" / "watch" / "gyro"
        if acc_c.exists() and gyr_c.exists():
            watch_acc_dir = acc_c
            watch_gyr_dir = gyr_c
            break

    if watch_acc_dir is None:
        # Give a helpful error showing what directories DO exist
        existing_dirs = sorted([str(p.relative_to(raw_dir)) for p in raw_dir.rglob("*") if p.is_dir()])[:20]
        raise FileNotFoundError(
            f"WISDM watch acc/gyro dirs not found under {raw_dir}.\n"
            f"  Expected structure: <root>/raw/watch/accel/ and <root>/raw/watch/gyro/\n"
            f"  Directories found under {raw_dir}:\n"
            + "\n".join(f"    {d}" for d in existing_dirs) +
            f"\n\n  HOW TO FIX:\n"
            f"  1. Check if the data has a double-nested folder (wisdm-dataset/wisdm-dataset/).\n"
            f"  2. Re-download from the assessment PDF link or run: bash setup_data.sh\n"
            f"  3. Ensure the zip extracts to: data/raw/wisdm/wisdm-dataset/raw/watch/accel/"
        )

    logger.info("  WISDM loading from %s", watch_acc_dir.parent)

    def parse_wisdm_file(filepath: Path) -> pd.DataFrame:
        """Parse WISDM raw sensor file: subject_id,activity,timestamp,x,y,z;"""
        rows = []
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                line = line.strip().rstrip(";").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                try:
                    rows.append({
                        "subject_id": int(parts[0]),
                        "activity_code": parts[1].strip(),
                        "timestamp": float(parts[2]),
                        "x": float(parts[3]),
                        "y": float(parts[4]),
                        "z": float(parts[5]),
                    })
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(rows)

    acc_dfs, gyr_dfs = {}, {}

    def _extract_subject_id_from_filename(fname: str) -> str:
        """Extract subject/user ID from WISDM filename.
        e.g. 'data_1600_accel_watch' -> '1600', 'data_1600_gyro_watch' -> '1600'
        """
        import re
        m = re.search(r"(\d{3,4})", fname)
        return m.group(1) if m else fname

    for f in sorted(watch_acc_dir.glob("*.txt")):
        df = parse_wisdm_file(f)
        if not df.empty:
            subj_key = _extract_subject_id_from_filename(f.stem)
            acc_dfs[subj_key] = df

    for f in sorted(watch_gyr_dir.glob("*.txt")):
        df = parse_wisdm_file(f)
        if not df.empty:
            subj_key = _extract_subject_id_from_filename(f.stem)
            gyr_dfs[subj_key] = df

    logger.info("  WISDM acc files: %d  gyro files: %d", len(acc_dfs), len(gyr_dfs))

    combined = []
    for key in sorted(acc_dfs):
        if key not in gyr_dfs:
            logger.debug("    No gyro file for acc key %s (no matching gyro), skipping", key)
            continue
        a = acc_dfs[key].sort_values("timestamp").reset_index(drop=True)
        g = gyr_dfs[key].sort_values("timestamp").reset_index(drop=True)

        # Merge on nearest timestamp (acc and gyro may be slightly misaligned)
        min_len = min(len(a), len(g))
        a = a.iloc[:min_len].copy()
        g = g.iloc[:min_len].copy()

        out = pd.DataFrame({
            "subject_id":   a["subject_id"].values,
            "activity_code": a["activity_code"].values,
            "acc_x": a["x"].values,
            "acc_y": a["y"].values,
            "acc_z": a["z"].values,
            "gyr_x": g["x"].values,
            "gyr_y": g["y"].values,
            "gyr_z": g["z"].values,
            "source_file": key,
        })
        combined.append(out)

    if not combined:
        raise RuntimeError("No WISDM data loaded.")

    result = pd.concat(combined, ignore_index=True)
    logger.info("  WISDM total rows: %d", len(result))
    return result


def preprocess_wisdm(raw_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    har_cfg = cfg["har"]
    source_hz = cfg["datasets"]["wisdm"]["source_rate_hz"]  # 20
    target_hz = har_cfg["target_rate_hz"]                   # 20  (no resampling needed)

    pretrain_ws   = int(har_cfg["pretrain"]["window_sec"] * target_hz)
    supervised_ws = int(har_cfg["supervised"]["window_sec"] * target_hz)
    supervised_step = int(supervised_ws * (1 - har_cfg["supervised"]["overlap_frac"]))

    label_map  = har_cfg["wisdm_label_map"]      # letter -> str
    unified_map = har_cfg["label_map"]            # str -> int

    df = load_wisdm(raw_dir, cfg)

    # Map to unified label
    df["label_str"] = df["activity_code"].map(label_map)
    df = df.dropna(subset=["label_str"])         # drop activities not in schema
    df["label_int"] = df["label_str"].map(unified_map)

    signal_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    max_gap = har_cfg["max_interpolation_gap_samples"]

    pretrain_signals, pretrain_meta = [], []
    sup_signals, sup_meta = [], []

    for subj in sorted(df["subject_id"].unique()):
        subj_df = df[df["subject_id"] == subj].copy()

        for col in signal_cols:
            arr = subj_df[col].values.astype(float)
            filled, ok = interpolate_missing(arr, max_gap)
            if not ok:
                series = pd.Series(arr).interpolate(limit=max_gap).ffill().bfill()
                filled = series.values
            subj_df[col] = filled

        sig = subj_df[signal_cols].values.T.astype(float)
        # No resampling needed: source already 20 Hz == target
        if source_hz != target_hz:
            sig = resample_signal(sig, source_hz, target_hz)

        label_arr  = subj_df["label_int"].values
        source_file = subj_df["source_file"].iloc[0]

        pt_wins  = window_signal(sig, pretrain_ws,   pretrain_ws)
        for w_idx, w in enumerate(pt_wins):
            pretrain_signals.append(w)
            pretrain_meta.append({
                "sample_id": make_sample_id("wisdm_pretrain", str(subj), str(source_file), w_idx),
                "dataset_name": "WISDM",
                "modality": "HAR",
                "subject_or_patient_id": str(subj),
                "source_file_or_record": str(source_file),
                "split": "pretrain",
                "label_or_event": None,
                "sampling_rate_hz": target_hz,
                "n_channels": w.shape[0],
                "n_samples": w.shape[1],
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "",
            })

        sup_wins = window_signal(sig, supervised_ws, supervised_step)
        n_new = sig.shape[1]
        for w_idx, w in enumerate(sup_wins):
            start = w_idx * supervised_step
            end   = start + supervised_ws
            win_labels = label_arr[start:min(end, len(label_arr))]
            win_labels_clean = [l for l in win_labels if not np.isnan(l)]
            if not win_labels_clean:
                continue
            majority_label = max(set(win_labels_clean), key=win_labels_clean.count)

            qc = []
            if np.any(np.isnan(w)):  qc.append("nan_present")
            if np.any(np.isinf(w)):  qc.append("inf_present")

            sup_signals.append(w)
            sup_meta.append({
                "sample_id": make_sample_id("wisdm_sup", str(subj), str(source_file), w_idx),
                "dataset_name": "WISDM",
                "modality": "HAR",
                "subject_or_patient_id": str(subj),
                "source_file_or_record": str(source_file),
                "split": "supervised",
                "label_or_event": int(majority_label),
                "sampling_rate_hz": target_hz,
                "n_channels": w.shape[0],
                "n_samples": w.shape[1],
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "|".join(qc),
            })

    pt_arr  = np.stack(pretrain_signals,  axis=0).astype(np.float32) if pretrain_signals else np.empty((0, 6, pretrain_ws), np.float32)
    sup_arr = np.stack(sup_signals, axis=0).astype(np.float32) if sup_signals else np.empty((0, 6, supervised_ws), np.float32)
    logger.info("  WISDM pretrain windows: %d  supervised windows: %d", len(pt_arr), len(sup_arr))
    return pt_arr, sup_arr, pretrain_meta, sup_meta


# ============================================================
# HAR: mHealth (BONUS)
# ============================================================

# mHealth columns (0-indexed):
# 0-2: chest acc, 3-4: ECG leads, 5-7: chest gyr, 8-10: chest mag
# 11-13: ankle acc, 14-16: ankle gyr, 17-19: ankle mag
# 20-22: wrist (right) acc, 23-25: wrist gyr, 26-28: wrist mag
# 29: label
MHEALTH_WRIST_ACC = [20, 21, 22]
MHEALTH_WRIST_GYR = [23, 24, 25]
MHEALTH_LABEL_COL = 29
MHEALTH_SOURCE_HZ = 50

def _load_mhealth_from_csv(csv_path: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """
    Fallback: load mHealth from a single CSV (Kaggle format).
    The CSV may have columns like: acc_chest_x, acc_chest_y, ..., label
    or may be headerless with 23 sensor columns + 1 label column.
    We attempt to parse it and map columns to the expected schema.
    """
    har_cfg = cfg["har"]
    target_hz = har_cfg["target_rate_hz"]  # 20
    pretrain_ws   = int(har_cfg["pretrain"]["window_sec"] * target_hz)
    supervised_ws = int(har_cfg["supervised"]["window_sec"] * target_hz)
    supervised_step = int(supervised_ws * (1 - har_cfg["supervised"]["overlap_frac"]))
    label_map_mh = har_cfg["mhealth_label_map"]
    unified_map  = har_cfg["label_map"]

    logger.info("  mHealth: loading from single CSV: %s", csv_path.name)

    # Try loading — detect if there's a header
    try:
        df_peek = pd.read_csv(csv_path, nrows=5)
        # If first row is numeric, likely no header
        if df_peek.columns[0].replace('.', '', 1).replace('-', '', 1).isdigit():
            df = pd.read_csv(csv_path, header=None, dtype=float)
        else:
            df = pd.read_csv(csv_path, dtype=float)
    except Exception as e:
        logger.error("  mHealth CSV load failed: %s", e)
        return (np.empty((0, 6, pretrain_ws), np.float32),
                np.empty((0, 6, supervised_ws), np.float32), [], [])

    logger.info("  mHealth CSV: shape %s, columns: %d", df.shape, df.shape[1])

    # The CSV might have a subject column or not. Check common patterns.
    # If there's a "subject" column, use it. Otherwise assign synthetic subject IDs
    # by splitting the data evenly into 10 parts (mHealth has 10 subjects).
    subject_col = None
    label_col_idx = None

    if df.shape[1] == 24:
        # Standard: 23 sensor columns + 1 label column (same as .log format)
        label_col_idx = 23
    elif df.shape[1] == 25:
        # 23 sensors + label + subject
        label_col_idx = 23
        subject_col = 24
    elif df.shape[1] >= 30:
        # Original format: 29 sensor cols + label (col 29 = label, 0-indexed)
        label_col_idx = 29
    else:
        # Last column is probably label
        label_col_idx = df.shape[1] - 1

    labels = df.iloc[:, label_col_idx].astype(int).values

    if subject_col is not None:
        subjects = df.iloc[:, subject_col].astype(int).values
    else:
        # Assign synthetic subject IDs: split into 10 equal parts
        n = len(df)
        subjects = np.repeat(np.arange(1, 11), n // 10 + 1)[:n]
        logger.warning("  mHealth CSV: no subject column found; assigning synthetic subject IDs (1-10)")

    # Extract wrist acc+gyr columns (6 channels)
    # In 24-col format: wrist acc might be at cols 14-16, gyr at 17-19
    # In 30-col format: wrist acc at cols 20-22, gyr at 23-25
    if df.shape[1] >= 30:
        acc_cols = [20, 21, 22]
        gyr_cols = [23, 24, 25]
    elif df.shape[1] >= 24:
        # Smaller format — try the last sensor columns before label
        # cols 0-2: chest acc, 3-5: chest gyr, 6-8: ankle acc, 9-11: ankle gyr
        # 12-14: wrist acc, 15-17: wrist gyr, 18-20: wrist mag, 21-22: ECG, 23: label
        acc_cols = [12, 13, 14]
        gyr_cols = [15, 16, 17]
    else:
        # Best guess: first 6 columns
        acc_cols = [0, 1, 2]
        gyr_cols = [3, 4, 5]
        logger.warning("  mHealth CSV: unknown column layout (%d cols). Using first 6 as acc+gyr.", df.shape[1])

    pretrain_signals, pretrain_meta = [], []
    sup_signals, sup_meta = [], []

    for subj_id in sorted(set(subjects)):
        mask = subjects == subj_id
        acc = df.iloc[mask, acc_cols].values.T  # [3, T_subj]
        gyr = df.iloc[mask, gyr_cols].values.T  # [3, T_subj]
        lbl = labels[mask]
        sig = np.vstack([acc, gyr])  # [6, T_subj]

        # Interpolate NaN
        for ch in range(6):
            sig[ch], _ = interpolate_missing(sig[ch], max_gap=10)

        # Resample from 50Hz to 20Hz
        sig = resample_signal(sig, MHEALTH_SOURCE_HZ, target_hz)
        ratio = target_hz / MHEALTH_SOURCE_HZ
        lbl_resampled = lbl[::int(1 / ratio)][:sig.shape[1]]

        # Normalize
        sig = zscore_normalize(sig)

        source = csv_path.name

        # Pretrain windows
        pt_windows = window_signal(sig, pretrain_ws, pretrain_ws)
        for w_idx, w in enumerate(pt_windows):
            qc = []
            if np.any(np.isnan(w)):
                qc.append("nan_present")
                w = np.nan_to_num(w, nan=0.0)
            pretrain_signals.append(w.astype(np.float32))
            pretrain_meta.append({
                "sample_id": make_sample_id("mhealth_pt", str(subj_id), source, w_idx),
                "dataset_name": "mHealth",
                "modality": "HAR",
                "subject_or_patient_id": str(subj_id),
                "source_file_or_record": source,
                "split": "pretrain",
                "label_or_event": None,
                "sampling_rate_hz": target_hz,
                "n_channels": 6,
                "n_samples": pretrain_ws,
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "|".join(qc),
            })

        # Supervised windows
        for w_idx in range(0, sig.shape[1] - supervised_ws + 1, supervised_step):
            w = sig[:, w_idx:w_idx + supervised_ws]
            w_labels = lbl_resampled[w_idx:w_idx + supervised_ws]

            # Exclude null label (0)
            valid = w_labels[w_labels != 0]
            if len(valid) == 0:
                continue

            majority_raw = int(pd.Series(valid).mode().iloc[0])
            mapped = label_map_mh.get(majority_raw)
            if mapped is None or mapped not in unified_map:
                continue
            majority_label = unified_map[mapped]

            qc = []
            if np.any(np.isnan(w)):
                qc.append("nan_present")
                w = np.nan_to_num(w, nan=0.0)

            sup_signals.append(w.astype(np.float32))
            sup_meta.append({
                "sample_id": make_sample_id("mhealth_sup", str(subj_id), source, w_idx),
                "dataset_name": "mHealth",
                "modality": "HAR",
                "subject_or_patient_id": str(subj_id),
                "source_file_or_record": source,
                "split": "supervised",
                "label_or_event": int(majority_label),
                "sampling_rate_hz": target_hz,
                "n_channels": 6,
                "n_samples": supervised_ws,
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "|".join(qc),
            })

    logger.info("  mHealth (CSV fallback): %d pretrain, %d supervised windows", len(pretrain_signals), len(sup_signals))

    pt_arr = np.stack(pretrain_signals, axis=0).astype(np.float32) if pretrain_signals else np.empty((0, 6, pretrain_ws), np.float32)
    sup_arr = np.stack(sup_signals, axis=0).astype(np.float32) if sup_signals else np.empty((0, 6, supervised_ws), np.float32)
    return pt_arr, sup_arr, pretrain_meta, sup_meta


def preprocess_mhealth(raw_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    har_cfg = cfg["har"]
    target_hz = har_cfg["target_rate_hz"]  # 20

    pretrain_ws   = int(har_cfg["pretrain"]["window_sec"] * target_hz)
    supervised_ws = int(har_cfg["supervised"]["window_sec"] * target_hz)
    supervised_step = int(supervised_ws * (1 - har_cfg["supervised"]["overlap_frac"]))

    label_map_mh = har_cfg["mhealth_label_map"]
    unified_map  = har_cfg["label_map"]

    # Find log files
    log_files = []
    for candidate_root in [raw_dir / "MHEALTHDATASET", raw_dir]:
        files = list(candidate_root.glob("mHealth_subject*.log"))
        if files:
            log_files = sorted(files)
            break

    if not log_files:
        # Check if user downloaded the Kaggle single-CSV version
        csv_files = list(raw_dir.glob("mhealth_raw_data.csv")) + list(raw_dir.glob("MHEALTHDATASET/*.csv"))
        if csv_files:
            logger.warning("  mHealth: found single CSV file instead of per-subject .log files.")
            logger.warning("  This is a Kaggle reformatted version. Attempting to parse it...")
            return _load_mhealth_from_csv(csv_files[0], cfg)
        raise FileNotFoundError(
            f"mHealth .log files not found under {raw_dir}.\n"
            f"  Expected: {raw_dir}/MHEALTHDATASET/mHealth_subject1.log ... mHealth_subject10.log\n"
            f"  The Kaggle version may have a different format.\n\n"
            f"  HOW TO FIX:\n"
            f"  1. Delete current files: rm data/raw/mhealth/*\n"
            f"  2. Download from Kaggle: https://www.kaggle.com/datasets/gaurav2022/mobile-health\n"
            f"     OR: https://www.kaggle.com/datasets/nirmalsankalana/mhealth-dataset-data-set\n"
            f"  3. Extract to data/raw/mhealth/ and re-run.\n"
            f"  If UCI is back online, just re-run: bash setup_data.sh"
        )

    logger.info("  mHealth loading %d subjects from %s", len(log_files), log_files[0].parent)

    pretrain_signals, pretrain_meta = [], []
    sup_signals, sup_meta = [], []

    for log_file in log_files:
        subj_id = int(log_file.stem.replace("mHealth_subject", ""))
        try:
            df = pd.read_csv(log_file, sep=r"\s+", header=None, dtype=float, na_values=["NaN"])
        except Exception as e:
            logger.warning("    mHealth: could not load %s: %s", log_file.name, e)
            continue

        if df.shape[1] < 30:
            logger.warning("    mHealth: unexpected columns %d in %s", df.shape[1], log_file.name)
            continue

        label_raw = df.iloc[:, MHEALTH_LABEL_COL].astype(int)
        # Map labels: label 0 (null) -> None -> drop
        label_mapped = label_raw.map(label_map_mh)
        df["label_str"] = label_mapped
        df = df.dropna(subset=["label_str"])
        df["label_int"] = df["label_str"].map(unified_map)
        df = df.dropna(subset=["label_int"])

        if df.empty:
            continue

        signal_cols_idx = MHEALTH_WRIST_ACC + MHEALTH_WRIST_GYR
        sig_raw = df.iloc[:, signal_cols_idx].values.T.astype(float)  # [6, T]

        # Resample 50 -> 20 Hz
        sig = resample_signal(sig_raw, MHEALTH_SOURCE_HZ, target_hz)
        label_arr = df["label_int"].values

        n_orig = sig_raw.shape[1]
        n_new  = sig.shape[1]
        ratio  = n_orig / n_new

        pt_wins = window_signal(sig, pretrain_ws, pretrain_ws)
        for w_idx, w in enumerate(pt_wins):
            pretrain_signals.append(w)
            pretrain_meta.append({
                "sample_id": make_sample_id("mhealth_pretrain", str(subj_id), "log", w_idx),
                "dataset_name": "mHealth",
                "modality": "HAR",
                "subject_or_patient_id": str(subj_id),
                "source_file_or_record": log_file.name,
                "split": "pretrain",
                "label_or_event": None,
                "sampling_rate_hz": target_hz,
                "n_channels": w.shape[0],
                "n_samples": w.shape[1],
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "",
            })

        sup_wins = window_signal(sig, supervised_ws, supervised_step)
        for w_idx, w in enumerate(sup_wins):
            start_orig = int(w_idx * supervised_step * ratio)
            end_orig   = int(start_orig + supervised_ws * ratio)
            win_labels = label_arr[start_orig:min(end_orig, len(label_arr))]
            win_labels_clean = [l for l in win_labels if not np.isnan(float(l))]
            if not win_labels_clean:
                continue
            majority_label = max(set(win_labels_clean), key=win_labels_clean.count)

            qc = []
            if np.any(np.isnan(w)):  qc.append("nan_present")
            if np.any(np.isinf(w)):  qc.append("inf_present")

            sup_signals.append(w)
            sup_meta.append({
                "sample_id": make_sample_id("mhealth_sup", str(subj_id), "log", w_idx),
                "dataset_name": "mHealth",
                "modality": "HAR",
                "subject_or_patient_id": str(subj_id),
                "source_file_or_record": log_file.name,
                "split": "supervised",
                "label_or_event": int(majority_label),
                "sampling_rate_hz": target_hz,
                "n_channels": w.shape[0],
                "n_samples": w.shape[1],
                "channel_schema": ",".join(har_cfg["channel_schema"]),
                "qc_flags": "|".join(qc),
            })

    pt_arr  = np.stack(pretrain_signals,  axis=0).astype(np.float32) if pretrain_signals else np.empty((0, 6, pretrain_ws), np.float32)
    sup_arr = np.stack(sup_signals, axis=0).astype(np.float32) if sup_signals else np.empty((0, 6, supervised_ws), np.float32)
    logger.info("  mHealth pretrain windows: %d  supervised windows: %d", len(pt_arr), len(sup_arr))
    return pt_arr, sup_arr, pretrain_meta, sup_meta


# ============================================================
# EEG: EEGMMIDB
# ============================================================

def preprocess_eegmmidb(raw_dir: Path, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """
    Load EDF+ motor imagery runs (4, 8, 12), parse T1/T2 annotations,
    bandpass + notch filter, common average reref, z-score normalise,
    extract fixed 4s event-aligned windows.

    Returns:
        signals [N, 64, 640]  (4s @ 160 Hz)
        metadata list[dict]
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required for EEG processing: pip install mne")

    eeg_cfg = cfg["eeg"]
    target_hz = eeg_cfg["target_rate_hz"]       # 160
    window_sec = eeg_cfg["window_sec"]           # 4.0
    window_samples = int(window_sec * target_hz) # 640
    runs = eeg_cfg["runs"]                       # [4, 8, 12]
    label_map = eeg_cfg["label_map"]             # T0/T1/T2 -> 0/1/2

    bp_lo, bp_hi = eeg_cfg["bandpass_hz"]
    notch_hz = eeg_cfg["notch_hz"]

    mne.set_log_level("WARNING")

    signals, meta = [], []
    n_subjects_ok = 0

    # Discover subject directories
    subj_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("S")])
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directories (S001, S002...) under {raw_dir}")

    logger.info("  EEGMMIDB: found %d subject dirs", len(subj_dirs))

    for subj_dir in subj_dirs:
        subj_id = subj_dir.name  # e.g. "S001"
        subj_ok = False

        for run in runs:
            run_str = f"{run:02d}"
            edf_path = subj_dir / f"{subj_id}R{run_str}.edf"

            if not edf_path.exists():
                continue

            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            except Exception as e:
                logger.warning("    EEG: could not load %s: %s", edf_path.name, e)
                continue

            # Resample if not at target (should already be 160 Hz)
            if int(raw.info["sfreq"]) != target_hz:
                raw.resample(target_hz, verbose=False)

            # Common average reference
            if eeg_cfg["reref"] == "average":
                raw.set_eeg_reference("average", projection=False, verbose=False)

            # Bandpass
            raw.filter(bp_lo, bp_hi, method="fir", verbose=False)

            # Notch (US 60 Hz)
            raw.notch_filter(notch_hz, verbose=False)

            # Get annotations
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Extract T1 and T2 events only (skip T0/rest)
            target_events = {}
            for ann_name, ev_id in event_id.items():
                ann_clean = ann_name.strip()
                if ann_clean in ("T1", "T2"):
                    target_events[ann_clean] = ev_id
                elif ann_clean in ("T0",):
                    pass  # T0 excluded by default

            if not target_events:
                logger.debug("    %s R%s: no T1/T2 events found, skipping", subj_id, run_str)
                continue

            data = raw.get_data()  # [64, T] in Volts

            # Detect and exclude corrupted segments (flat or extreme amplitude channels)
            channel_ranges = data.max(axis=1) - data.min(axis=1)
            flat_channels = np.where(channel_ranges < 1e-10)[0]
            corrupt_channels = np.where(channel_ranges > 500e-6)[0]  # >500 µV is artefactual
            bad_ch_mask = np.zeros(data.shape[0], dtype=bool)
            bad_ch_mask[flat_channels] = True
            bad_ch_mask[corrupt_channels] = True

            for ann_name, ev_id in target_events.items():
                ev_mask = events[:, 2] == ev_id
                ev_samples = events[ev_mask, 0]

                for ev_idx, onset in enumerate(ev_samples):
                    end = onset + window_samples
                    if end > data.shape[1]:
                        continue

                    window = data[:, onset:end].copy()  # [64, 640]

                    # Replace bad channels with NaN, mark in QC
                    qc = []
                    if bad_ch_mask.any():
                        window[bad_ch_mask, :] = np.nan
                        qc.append(f"bad_ch:{np.sum(bad_ch_mask)}")

                    # Check NaN/Inf
                    if np.any(np.isnan(window)):
                        qc.append("nan_present")
                    if np.any(np.isinf(window)):
                        qc.append("inf_present")
                        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

                    # Z-score per channel (skip NaN channels)
                    for ch_i in range(window.shape[0]):
                        if not np.all(np.isnan(window[ch_i])):
                            ch_mean = np.nanmean(window[ch_i])
                            ch_std  = np.nanstd(window[ch_i])
                            ch_std  = ch_std if ch_std > 1e-10 else 1.0
                            window[ch_i] = (window[ch_i] - ch_mean) / ch_std

                    # Replace any remaining NaN (from bad channels) with 0.0
                    # NaN channels have already been flagged in qc_flags above
                    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

                    signals.append(window.astype(np.float32))
                    meta.append({
                        "sample_id": f"eeg_{subj_id}_R{run_str}_{ann_name}_ev{ev_idx:03d}",
                        "dataset_name": "EEGMMIDB",
                        "modality": "EEG",
                        "subject_or_patient_id": subj_id,
                        "source_file_or_record": edf_path.name,
                        "run_id": run_str,
                        "split": "supervised",
                        "label_or_event": label_map[ann_name],
                        "sampling_rate_hz": target_hz,
                        "n_channels": window.shape[0],
                        "n_samples": window.shape[1],
                        "channel_schema": "10-10_system_64ch",
                        "qc_flags": "|".join(qc),
                    })
                    subj_ok = True

        if subj_ok:
            n_subjects_ok += 1

    logger.info("  EEGMMIDB: %d subjects processed, %d windows extracted", n_subjects_ok, len(signals))

    if not signals:
        raise RuntimeError("No EEG windows extracted. Check EDF files.")

    arr = np.stack(signals, axis=0).astype(np.float32)
    return arr, meta


# ============================================================
# ECG: PTB-XL
# ============================================================

def preprocess_ptbxl(raw_dir: Path, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    """
    Load PTB-XL using wfdb.
    Returns:
        signals [N, 12, T]   (T = 1000 @ 100Hz or 5000 @ 500Hz)
        metadata list[dict]
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is required for ECG processing: pip install wfdb")

    ecg_cfg = cfg["ecg"]
    target_hz = ecg_cfg["target_rate_hz"]  # 100

    # Find ptbxl root
    ptbxl_root = None
    for candidate in [raw_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
                       raw_dir]:
        if (candidate / "ptbxl_database.csv").exists():
            ptbxl_root = candidate
            break

    if ptbxl_root is None:
        raise FileNotFoundError(
            f"ptbxl_database.csv not found under {raw_dir}. "
            "Ensure PTB-XL has been downloaded and extracted."
        )

    logger.info("  PTB-XL loading metadata from %s", ptbxl_root)

    # Load metadata
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv", index_col="ecg_id")
    scp_df = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)

    # Map SCP codes to superclass labels
    scp_diag = scp_df[scp_df["diagnostic"] == 1.0]
    superclass_map = scp_diag["diagnostic_class"].to_dict()

    def get_superclass(scp_codes_str: str) -> str | None:
        """Parse the SCP codes string dict and return the dominant superclass."""
        try:
            import ast
            scp_dict = ast.literal_eval(scp_codes_str)
        except Exception:
            return None
        best_label, best_conf = None, -1
        for code, conf in scp_dict.items():
            if code in superclass_map and conf > best_conf:
                best_label, best_conf = superclass_map[code], conf
        return best_label

    db["superclass"] = db["scp_codes"].apply(get_superclass)
    db = db.dropna(subset=["superclass"])

    # Patient-level splits using strat_fold
    train_folds = ecg_cfg["train_folds"]
    val_folds   = ecg_cfg["val_folds"]
    test_folds  = ecg_cfg["test_folds"]

    def assign_split(fold):
        if fold in train_folds:  return "train"
        if fold in val_folds:    return "val"
        if fold in test_folds:   return "test"
        return "unknown"

    db["split"] = db["strat_fold"].apply(assign_split)
    db = db[db["split"] != "unknown"]

    logger.info("  PTB-XL records: %d  (train=%d val=%d test=%d)",
                len(db),
                (db["split"] == "train").sum(),
                (db["split"] == "val").sum(),
                (db["split"] == "test").sum())

    # Choose correct waveform path column
    if target_hz == 100:
        path_col = "filename_lr"
    else:
        path_col = "filename_hr"

    bp_lo, bp_hi = ecg_cfg["bandpass_hz"]
    lead_names = ecg_cfg["lead_names"]

    signals, meta = [], []
    n_errors = 0
    n_total  = len(db)

    logger.info("  PTB-XL: loading %d records at %d Hz ...", n_total, target_hz)

    for ecg_id, row in db.iterrows():
        rec_path = ptbxl_root / row[path_col]
        try:
            rec = wfdb.rdrecord(str(rec_path.with_suffix("")))
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                logger.warning("    PTB-XL: could not load %s: %s", rec_path, e)
            continue

        sig = rec.p_signal.T.astype(float)  # [12, T]

        # Bandpass filter
        sig = bandpass_filter(sig, bp_lo, bp_hi, fs=target_hz)

        # Per-record z-score
        sig = zscore_normalize(sig)

        # QC
        qc = []
        if np.any(np.isnan(sig)):  qc.append("nan_present")
        if np.any(np.isinf(sig)):  qc.append("inf_present")
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        signals.append(sig.astype(np.float32))
        meta.append({
            "sample_id": f"ptbxl_{ecg_id:05d}",
            "dataset_name": "PTB-XL",
            "modality": "ECG",
            "subject_or_patient_id": str(row.get("patient_id", "")),
            "source_file_or_record": row[path_col],
            "split": row["split"],
            "label_or_event": row["superclass"],
            "sampling_rate_hz": target_hz,
            "n_channels": sig.shape[0],
            "n_samples": sig.shape[1],
            "channel_schema": ",".join(lead_names),
            "qc_flags": "|".join(qc),
            "strat_fold": int(row["strat_fold"]),
            "age": row.get("age", ""),
            "sex": row.get("sex", ""),
        })

    logger.info("  PTB-XL: loaded %d / %d records (%d errors)",
                len(signals), n_total, n_errors)

    if not signals:
        raise RuntimeError("No PTB-XL records loaded.")

    arr = np.stack(signals, axis=0).astype(np.float32)
    return arr, meta


# ============================================================
# Main orchestration
# ============================================================

def run_har(cfg: dict, out_dir: Path, submission_dir: Path) -> list[dict]:
    """Process all HAR datasets and save outputs."""
    logger.info("=" * 60)
    logger.info("HAR PREPROCESSING")
    logger.info("=" * 60)

    raw_base = Path(cfg["paths"]["raw"])
    all_pretrain, all_pretrain_meta = [], []
    all_sup, all_sup_meta = [], []
    file_manifest = []

    # PAMAP2
    logger.info("[HAR] Processing PAMAP2 ...")
    pt, sup, pt_meta, sup_meta = preprocess_pamap2(raw_base / "pamap2", cfg)
    all_pretrain.append(pt);    all_pretrain_meta.extend(pt_meta)
    all_sup.append(sup);        all_sup_meta.extend(sup_meta)

    # WISDM
    logger.info("[HAR] Processing WISDM ...")
    pt, sup, pt_meta, sup_meta = preprocess_wisdm(raw_base / "wisdm", cfg)
    all_pretrain.append(pt);    all_pretrain_meta.extend(pt_meta)
    all_sup.append(sup);        all_sup_meta.extend(sup_meta)

    # mHealth (bonus)
    mhealth_raw = raw_base / "mhealth"
    if any(mhealth_raw.rglob("*.log")):
        logger.info("[HAR] Processing mHealth (bonus) ...")
        try:
            pt, sup, pt_meta, sup_meta = preprocess_mhealth(mhealth_raw, cfg)
            all_pretrain.append(pt);    all_pretrain_meta.extend(pt_meta)
            all_sup.append(sup);        all_sup_meta.extend(sup_meta)
        except Exception as e:
            logger.warning("[HAR] mHealth preprocessing failed: %s", e)
    else:
        logger.info("[HAR] mHealth not found, skipping bonus dataset.")

    # Concatenate
    pretrain_arr = np.concatenate([a for a in all_pretrain if a.size > 0], axis=0)
    sup_arr = np.concatenate([a for a in all_sup if a.size > 0], axis=0)

    logger.info("[HAR] Total pretrain windows: %d  supervised windows: %d",
                len(pretrain_arr), len(sup_arr))

    # Save
    pt_path  = out_dir / "har_pretrain.npz"
    sup_path = out_dir / "har_supervised.npz"
    save_npz(pt_path,  pretrain_arr, all_pretrain_meta)
    save_npz(sup_path, sup_arr, all_sup_meta)

    # Submission sample (100 from each)
    for src_arr, src_meta, name in [
        (pretrain_arr, all_pretrain_meta, "har_pretrain_sample"),
        (sup_arr, all_sup_meta, "har_supervised_sample"),
    ]:
        idx = np.random.choice(len(src_arr), min(100, len(src_arr)), replace=False)
        samp = src_arr[idx]
        samp_meta = [src_meta[i] for i in idx]
        save_npz(submission_dir / f"{name}.npz", samp, samp_meta)

    for path, arr, meta in [
        (pt_path, pretrain_arr, all_pretrain_meta),
        (sup_path, sup_arr, all_sup_meta),
    ]:
        file_manifest.append({
            "file": str(path),
            "modality": "HAR",
            "shape": list(arr.shape),
            "n_windows": len(arr),
            "n_metadata_rows": len(meta),
            "size_bytes": path.stat().st_size,
        })

    return file_manifest


def run_eeg(cfg: dict, out_dir: Path, submission_dir: Path) -> list[dict]:
    logger.info("=" * 60)
    logger.info("EEG PREPROCESSING")
    logger.info("=" * 60)

    raw_dir = Path(cfg["paths"]["raw"]) / "eegmmidb"
    if not any(raw_dir.rglob("*.edf")):
        logger.warning("[EEG] No EDF files found in %s. Skipping EEG.", raw_dir)
        return []

    arr, meta = preprocess_eegmmidb(raw_dir, cfg)

    out_path = out_dir / "eeg_motor_imagery.npz"
    save_npz(out_path, arr, meta)

    # Submission sample
    idx = np.random.choice(len(arr), min(100, len(arr)), replace=False)
    save_npz(submission_dir / "eeg_motor_imagery_sample.npz", arr[idx], [meta[i] for i in idx])

    return [{
        "file": str(out_path),
        "modality": "EEG",
        "shape": list(arr.shape),
        "n_windows": len(arr),
        "n_metadata_rows": len(meta),
        "size_bytes": out_path.stat().st_size,
    }]


def run_ecg(cfg: dict, out_dir: Path, submission_dir: Path) -> list[dict]:
    logger.info("=" * 60)
    logger.info("ECG PREPROCESSING")
    logger.info("=" * 60)

    raw_dir = Path(cfg["paths"]["raw"]) / "ptbxl"
    arr, meta = preprocess_ptbxl(raw_dir, cfg)

    out_path = out_dir / "ecg_ptbxl.npz"
    save_npz(out_path, arr, meta)

    # Submission sample — stratified by split
    meta_df = pd.DataFrame(meta)
    sample_idx = []
    for split in ["train", "val", "test"]:
        split_idx = meta_df.index[meta_df["split"] == split].tolist()
        n = min(34, len(split_idx))
        sample_idx.extend(np.random.choice(split_idx, n, replace=False).tolist())
    sample_idx = sample_idx[:100]
    save_npz(submission_dir / "ecg_ptbxl_sample.npz",
             arr[sample_idx], [meta[i] for i in sample_idx])

    return [{
        "file": str(out_path),
        "modality": "ECG",
        "shape": list(arr.shape),
        "n_windows": len(arr),
        "n_metadata_rows": len(meta),
        "size_bytes": out_path.stat().st_size,
    }]


def write_processed_manifest(entries: list[dict], out_path: Path) -> None:
    manifest = {
        "manifest_type": "processed_manifest",
        "generated_at": datetime.now(tz=None).astimezone().isoformat(),
        "pipeline_version": "1.0.0",
        "files": entries,
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Processed manifest written to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Multimodal SSL preprocessing pipeline")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--modality", choices=["har", "eeg", "ecg", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg = load_config(args.config)

    proc_base = Path(cfg["paths"]["processed"])
    sub_base  = Path(cfg["paths"]["submission_sample"])
    rep_dir   = Path(cfg["paths"]["reports"])
    ensure_dir(proc_base / "har")
    ensure_dir(proc_base / "eeg")
    ensure_dir(proc_base / "ecg")
    ensure_dir(sub_base  / "har")
    ensure_dir(sub_base  / "eeg")
    ensure_dir(sub_base  / "ecg")
    ensure_dir(rep_dir)

    t0 = time.time()
    all_manifest_entries = []

    if args.modality in ("har", "all"):
        entries = run_har(cfg, proc_base / "har", sub_base / "har")
        all_manifest_entries.extend(entries)

    if args.modality in ("eeg", "all"):
        entries = run_eeg(cfg, proc_base / "eeg", sub_base / "eeg")
        all_manifest_entries.extend(entries)

    if args.modality in ("ecg", "all"):
        entries = run_ecg(cfg, proc_base / "ecg", sub_base / "ecg")
        all_manifest_entries.extend(entries)

    write_processed_manifest(all_manifest_entries, rep_dir / "processed_manifest.json")

    elapsed = time.time() - t0
    logger.info("Preprocessing complete in %.1f seconds", elapsed)
    logger.info("Run validation: python validate_outputs.py --config %s", args.config)


if __name__ == "__main__":
    main()
