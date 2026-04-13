#!/usr/bin/env python3
"""
tests/test_pipeline.py
======================
Smoke tests + unit tests for the preprocessing pipeline.
Run with:  python -m pytest tests/ -v
or:        python tests/test_pipeline.py
"""

import sys
import os
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import (
    resample_signal,
    bandpass_filter,
    notch_filter,
    zscore_normalize,
    interpolate_missing,
    window_signal,
    make_sample_id,
    save_npz,
)


# ============================================================
# Unit tests: signal processing utilities
# ============================================================

class TestResampleSignal(unittest.TestCase):

    def test_identity(self):
        """No-op when src==target rate."""
        sig = np.random.randn(6, 200).astype(float)
        out = resample_signal(sig, 100, 100)
        np.testing.assert_array_equal(sig, out)

    def test_downsample_100_to_20(self):
        """100 Hz -> 20 Hz: output length should be 1/5."""
        sig = np.random.randn(6, 1000).astype(float)
        out = resample_signal(sig, 100, 20)
        self.assertEqual(out.shape[0], 6)
        self.assertEqual(out.shape[1], 200)

    def test_downsample_50_to_20(self):
        """50 Hz -> 20 Hz: output length should be 2/5."""
        sig = np.random.randn(6, 1000).astype(float)
        out = resample_signal(sig, 50, 20)
        self.assertEqual(out.shape[1], 400)

    def test_output_channels_preserved(self):
        """Channel count must not change."""
        sig = np.random.randn(3, 500).astype(float)
        out = resample_signal(sig, 100, 20)
        self.assertEqual(out.shape[0], 3)


class TestBandpassFilter(unittest.TestCase):

    def test_output_shape(self):
        sig = np.random.randn(6, 400).astype(float)
        out = bandpass_filter(sig, 1.0, 40.0, fs=100)
        self.assertEqual(out.shape, sig.shape)

    def test_no_nan(self):
        sig = np.random.randn(6, 400).astype(float)
        out = bandpass_filter(sig, 0.5, 40.0, fs=100)
        self.assertFalse(np.any(np.isnan(out)))

    def test_attenuates_high_freq(self):
        """80 Hz content should be attenuated below a 40 Hz cutoff at 500 Hz sample rate."""
        # Use enough samples for filtfilt padding (needs >3*filter_order)
        t = np.linspace(0, 4, 2000)   # 4s at 500 Hz
        high_freq = np.sin(2 * np.pi * 80 * t)
        low_freq  = np.sin(2 * np.pi * 10 * t)
        # Mix: 10 Hz should pass, 80 Hz should be attenuated
        sig = (high_freq + low_freq)[np.newaxis, :]
        out = bandpass_filter(sig, 1.0, 40.0, fs=500, order=4)
        # After filtering, 80 Hz component is largely removed
        # Residual power should be dominated by 10 Hz, not 80 Hz
        self.assertLess(out.std(), sig.std())  # output should have less energy than noisy input


class TestNotchFilter(unittest.TestCase):

    def test_output_shape(self):
        sig = np.random.randn(64, 640).astype(float)
        out = notch_filter(sig, freq=60.0, fs=160)
        self.assertEqual(out.shape, sig.shape)


class TestZscoreNormalize(unittest.TestCase):

    def test_mean_near_zero(self):
        sig = np.random.randn(6, 200).astype(float) * 10 + 5
        out = zscore_normalize(sig)
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros(6), atol=1e-5)

    def test_std_near_one(self):
        sig = np.random.randn(6, 200).astype(float) * 10 + 5
        out = zscore_normalize(sig)
        np.testing.assert_allclose(out.std(axis=-1), np.ones(6), atol=0.01)

    def test_flat_channel_no_crash(self):
        """Flat channel should not cause division by zero."""
        sig = np.ones((6, 200)).astype(float)
        sig[0] = 5.0  # all constant
        out = zscore_normalize(sig)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))


class TestInterpolateMissing(unittest.TestCase):

    def test_no_nans(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        filled, ok = interpolate_missing(arr, max_gap=5)
        np.testing.assert_array_equal(filled, arr)
        self.assertTrue(ok)

    def test_small_gap_filled(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0])
        filled, ok = interpolate_missing(arr, max_gap=5)
        self.assertTrue(ok)
        self.assertFalse(np.any(np.isnan(filled)))
        # Linear interpolation: 2.0, 3.0
        np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 4.0], atol=0.01)

    def test_large_gap_flagged(self):
        arr = np.array([1.0] + [np.nan] * 15 + [4.0])
        _, ok = interpolate_missing(arr, max_gap=10)
        self.assertFalse(ok)


class TestWindowSignal(unittest.TestCase):

    def test_basic(self):
        sig = np.random.randn(6, 400)
        windows = window_signal(sig, window_samples=200, step_samples=200)
        self.assertEqual(windows.shape, (2, 6, 200))

    def test_overlap(self):
        sig = np.random.randn(6, 400)
        # step = 100 (50% overlap on 200-sample window)
        windows = window_signal(sig, window_samples=200, step_samples=100)
        self.assertEqual(windows.shape, (3, 6, 200))

    def test_signal_too_short(self):
        sig = np.random.randn(6, 50)
        windows = window_signal(sig, window_samples=200, step_samples=200)
        self.assertEqual(windows.shape[0], 0)

    def test_content_correct(self):
        """First window should equal first 200 samples."""
        sig = np.arange(6 * 200).reshape(6, 200).astype(float)
        sig_long = np.concatenate([sig, sig], axis=1)  # [6, 400]
        windows = window_signal(sig_long, 200, 200)
        np.testing.assert_array_equal(windows[0], sig)


class TestSampleId(unittest.TestCase):

    def test_format(self):
        sid = make_sample_id("pamap2_sup", "3", "subject3.dat", 42)
        self.assertIn("pamap2_sup", sid)
        self.assertIn("sub3", sid)
        self.assertIn("w00042", sid)


# ============================================================
# Smoke tests: save/load roundtrip
# ============================================================

class TestNpzRoundtrip(unittest.TestCase):

    def test_save_load(self):
        from preprocess import save_npz
        from validate_outputs import load_npz

        signals = np.random.randn(5, 6, 200).astype(np.float32)
        meta = [
            {
                "sample_id": f"test_{i}",
                "dataset_name": "TEST",
                "modality": "HAR",
                "subject_or_patient_id": "1",
                "source_file_or_record": "test.dat",
                "split": "train",
                "label_or_event": i % 3,
                "sampling_rate_hz": 20,
                "n_channels": 6,
                "n_samples": 200,
                "channel_schema": "acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z",
                "qc_flags": "",
            }
            for i in range(5)
        ]

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            p = Path(tmp.name)

        try:
            save_npz(p, signals, meta)
            loaded_sig, loaded_meta = load_npz(p)
            np.testing.assert_array_equal(signals, loaded_sig)
            self.assertEqual(len(loaded_meta), 5)
            self.assertIn("sample_id", loaded_meta.columns)
        finally:
            p.unlink(missing_ok=True)


# ============================================================
# Format / manifest smoke tests
# ============================================================

class TestManifestFormat(unittest.TestCase):

    def test_download_manifest_parseable(self):
        """Validate that the download manifest JSON (if it exists) is valid."""
        manifest_path = Path("reports/download_manifest.json")
        if not manifest_path.exists():
            self.skipTest("Download manifest not yet generated")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertIn("entries", manifest)
        self.assertIsInstance(manifest["entries"], list)

    def test_processed_manifest_parseable(self):
        manifest_path = Path("reports/processed_manifest.json")
        if not manifest_path.exists():
            self.skipTest("Processed manifest not yet generated")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertIn("files", manifest)
        for entry in manifest["files"]:
            self.assertIn("file", entry)
            self.assertIn("shape", entry)
            self.assertIn("n_windows", entry)

    def test_validation_report_parseable(self):
        rep_path = Path("reports/validation_report.json")
        if not rep_path.exists():
            self.skipTest("Validation report not yet generated")
        with open(rep_path) as f:
            report = json.load(f)
        self.assertIn("summary", report)
        self.assertIn("checks", report)
        self.assertIn("resource_estimate", report)


# ============================================================
# HAR label schema smoke tests
# ============================================================

class TestLabelSchema(unittest.TestCase):

    def _load_cfg(self):
        import yaml
        cfg_path = Path("configs/pipeline_config.yaml")
        if not cfg_path.exists():
            self.skipTest("Config not found")
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    def test_pamap2_label_values_in_unified_map(self):
        cfg = self._load_cfg()
        unified = set(cfg["har"]["label_map"].keys())
        for k, v in cfg["har"]["pamap2_label_map"].items():
            self.assertIn(v, unified,
                          f"PAMAP2 label {k}->{v} not in unified map")

    def test_wisdm_label_values_in_unified_map(self):
        cfg = self._load_cfg()
        unified = set(cfg["har"]["label_map"].keys())
        for k, v in cfg["har"]["wisdm_label_map"].items():
            if v is not None:
                self.assertIn(v, unified,
                              f"WISDM label {k}->{v} not in unified map")

    def test_unified_label_indices_contiguous(self):
        cfg = self._load_cfg()
        values = sorted(cfg["har"]["label_map"].values())
        self.assertEqual(values, list(range(len(values))),
                         "Unified label indices should be 0,1,2,...")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
