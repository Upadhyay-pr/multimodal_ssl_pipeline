# Multimodal SSL Preprocessing Pipeline

**Assessment:** MED05664 — Research Assistant in Bioinformatics, Imperial College London  
**Candidate:** Pragati Upadhyay  
**Submission date:** April 2026  

---

## What this repository does

This pipeline downloads, organises, preprocesses, and validates five open
multimodal time-series datasets for downstream self-supervised learning (SSL).
The output is a set of fixed-shape float32 arrays (`.npz` files) ready to be
loaded into any deep-learning framework.

| Dataset | Modality | Status | Size |
|---------|----------|--------|------|
| PAMAP2 | HAR — wrist IMU (100 Hz) | **Mandatory** | ~500 MB |
| WISDM | HAR — watch acc+gyro (20 Hz) | **Mandatory** | ~250 MB |
| mHealth | HAR + 2-lead ECG (50 Hz) | **Bonus** | ~40 MB |
| EEGMMIDB | EEG — 64-channel motor imagery (160 Hz) | **Mandatory** | ~3 GB (runs 4,8,12 only) |
| PTB-XL | ECG — 12-lead clinical (100 Hz) | **Mandatory** | ~1.7 GB |

---

## Repository structure

```
multimodal_ssl_pipeline/
├── setup_data.sh              # Shell wrapper: venv + folders + downloads
├── setup.bat                  # Windows equivalent of setup_data.sh
├── download_datasets.py       # Pure-Python downloader (fallback / cross-platform)
├── preprocess.py              # Ingestion, preprocessing, output generation
├── validate_outputs.py        # Validation checks + resource report
├── configs/
│   └── pipeline_config.yaml  # All parameters (rates, windows, label maps)
├── data/                      # ← not committed to git (created by setup_data.sh)
│   ├── raw/                   #   Downloaded archives
│   ├── interim/               #   Intermediate files
│   └── processed/             #   Final .npz arrays
├── reports/
│   ├── preprocessing_plan.md  # One-page design document (required deliverable)
│   ├── ssl_downstream_note.md # Bonus: how outputs feed an SSL pipeline
│   ├── download_manifest.json # Auto-generated: download provenance
│   ├── processed_manifest.json# Auto-generated: processed file inventory
│   ├── validation_report.json # Auto-generated: machine-readable validation
│   └── validation_report.txt  # Auto-generated: human-readable validation
├── submission_sample/         # 100 representative samples per dataset (committed)
│   ├── har/
│   ├── eeg/
│   └── ecg/
├── tests/
│   └── test_pipeline.py      # Unit + smoke tests (pytest-compatible)
├── venv/                      # ← not committed (created by setup_data.sh)
├── requirements.txt
└── README.md
```

---

## Quickstart — three commands

```bash
# 1. Download data and set up virtual environment
bash setup_data.sh

# 2. Preprocess all modalities
source venv/bin/activate          # Windows: venv\Scripts\activate
python preprocess.py --config configs/pipeline_config.yaml

# 3. Validate outputs and generate report
python validate_outputs.py --config configs/pipeline_config.yaml
```

That is all that is needed to reproduce the full pipeline from scratch.

---

## Detailed step-by-step instructions

### Prerequisites

| Requirement | Minimum version | Check command |
|-------------|-----------------|---------------|
| Python | 3.8+ | `python3 --version` |
| bash | 3.2+ | `bash --version` |
| wget **or** curl | any | `wget --version` or `curl --version` |
| unzip | any | `unzip -v` |

**macOS:** All prerequisites are available via Homebrew.
```bash
brew install python wget   # wget optional — curl is used automatically if absent
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y python3 python3-venv wget unzip
```

**Windows:** Use Git Bash (recommended), WSL (Ubuntu), or see `setup.bat`.

---

### Step 0 — Clone the repository

```bash
git clone https://github.com/Upadhyay-pr/multimodal_ssl_pipeline.git
cd multimodal_ssl_pipeline
```

---

### Step 1 — Download datasets and create virtual environment

The `setup_data.sh` script does **four things only**:
1. Creates the directory tree (`data/`, `reports/`, `submission_sample/`)
2. Creates a Python virtual environment (`venv/`) and installs `requirements.txt`
3. Downloads all raw datasets
4. Writes a machine-readable `reports/download_manifest.json`

```bash
# Full pipeline — all datasets (recommended)
bash setup_data.sh

# Skip the large EEG download (~3 GB) to test faster
bash setup_data.sh --skip-eeg

# Skip PTB-XL (~1.7 GB) as well
bash setup_data.sh --skip-eeg --skip-ecg

# Skip bonus mHealth dataset
bash setup_data.sh --no-mhealth

# Dry run: print what would be downloaded without downloading anything
bash setup_data.sh --dry-run

# Skip venv creation (if you manage your own environment)
bash setup_data.sh --skip-venv

# Show all options
bash setup_data.sh --help
```

> **Note — macOS without wget:**
> `curl` is used automatically. For PhysioNet recursive downloads (PTB-XL),
> the Python fallback downloader (`download_datasets.py`) is called automatically.
> No manual action needed.

> **Note — Windows Git Bash:**
> Run `bash setup_data.sh` inside Git Bash. All paths use forward slashes
> and the script detects Windows automatically.
> Alternatively, use `setup.bat` (see Windows section below).

> **Downloads can be interrupted and resumed.** Re-running `setup_data.sh`
> skips files that are already downloaded (checked by file size).

---

### Step 2 — Activate the virtual environment

```bash
# macOS / Linux
source venv/bin/activate

# Windows Git Bash
source venv/Scripts/activate

# Windows Command Prompt
venv\Scripts\activate.bat

# Windows PowerShell
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt. All `python` commands below
assume the venv is active.

---

### Step 3 — Run preprocessing

```bash
# Process all four modalities (HAR + EEG + ECG)
python preprocess.py --config configs/pipeline_config.yaml

# Process one modality at a time (useful for debugging)
python preprocess.py --config configs/pipeline_config.yaml --modality har
python preprocess.py --config configs/pipeline_config.yaml --modality eeg
python preprocess.py --config configs/pipeline_config.yaml --modality ecg
```

**What preprocessing does:**

| Modality | Input | Operations | Output |
|----------|-------|-----------|--------|
| HAR | PAMAP2 (100 Hz) + WISDM (20 Hz) | Resample → 20 Hz, z-score, window | `har_pretrain.npz` [N, 6, 200], `har_supervised.npz` [N, 6, 100] |
| EEG | EEGMMIDB EDF+ (160 Hz) | CAR rereference, 1–40 Hz bandpass, 60 Hz notch, event windows | `eeg_motor_imagery.npz` [N, 64, 640] |
| ECG | PTB-XL WFDB (100 Hz) | 0.5–40 Hz bandpass, z-score, patient splits | `ecg_ptbxl.npz` [N, 12, 1000] |

Expected runtime: 45–90 min single-core (dominated by EEG and ECG loading).

---

### Step 4 — Validate outputs

```bash
# Generate validation report + resource estimate
python validate_outputs.py --config configs/pipeline_config.yaml

# Strict mode: exits with code 1 if any check fails (useful in CI)
python validate_outputs.py --config configs/pipeline_config.yaml --strict
```

The script checks:
1. All output files exist and are non-empty
2. No silent NaNs or Inf values in any array
3. Float32 dtype throughout
4. HAR: PAMAP2 and WISDM both at 20 Hz with identical 6-channel schema
5. HAR window lengths match specification (pretrain=200, supervised=100 samples)
6. No null labels in supervised output (PAMAP2 class 0 excluded)
7. Subject/patient IDs preserved (leakage control)
8. No patient in both ECG train and test splits
9. EEG labels are T1/T2 only; run IDs present
10. PTB-XL strat_fold metadata; all splits (train/val/test) present
11. Submission sample packs ≥ 10 windows each

Reports saved to:
- `reports/validation_report.json` — machine-readable
- `reports/validation_report.txt` — human-readable

---

### Step 5 — Run tests

```bash
# Run all unit + smoke tests
python -m pytest tests/ -v

# Run with short output
python -m pytest tests/ -q

# Run just the signal processing unit tests
python -m pytest tests/ -v -k "TestResample or TestBandpass or TestZscore"

# Run without pytest (if not installed)
python tests/test_pipeline.py
```

---

## Alternative: Python-only download (no bash required)

If you cannot run bash (e.g. Windows without Git Bash), use the pure-Python downloader:

```bash
# Full download
python download_datasets.py

# Individual datasets
python download_datasets.py --dataset pamap2
python download_datasets.py --dataset wisdm
python download_datasets.py --dataset eegmmidb   # ~3 GB
python download_datasets.py --dataset ptbxl      # ~1.7 GB

# List available datasets
python download_datasets.py --list

# Dry run
python download_datasets.py --dry-run
```

---

## Windows instructions

### Option A: Git Bash (recommended)

1. Install [Git for Windows](https://gitforwindows.org/) — includes Git Bash and curl
2. Open **Git Bash**
3. Run `bash setup_data.sh` as described above

### Option B: WSL (Windows Subsystem for Linux)

1. Enable WSL: `wsl --install` in PowerShell (admin)
2. Open Ubuntu terminal from Start menu
3. Clone the repo and run `bash setup_data.sh`

### Option C: Windows batch file

```bat
REM Double-click setup.bat or run from Command Prompt:
setup.bat

REM Then activate the environment:
venv\Scripts\activate.bat

REM Then run preprocessing:
python preprocess.py --config configs\pipeline_config.yaml
```

---

## Outputs

### Processed arrays (`.npz`)

All outputs are stored as compressed NumPy archives with shape `[N, C, T]`
(batch × channels × time samples), dtype `float32`.

| File | Shape | Description |
|------|-------|-------------|
| `data/processed/har/har_pretrain.npz` | [N, 6, 200] | 10s HAR windows — no labels (SSL pretrain) |
| `data/processed/har/har_supervised.npz` | [N, 6, 100] | 5s HAR windows, 50% overlap — labelled |
| `data/processed/eeg/eeg_motor_imagery.npz` | [N, 64, 640] | 4s EEG event-aligned windows |
| `data/processed/ecg/ecg_ptbxl.npz` | [N, 12, 1000] | 10s 12-lead ECG records |

### Loading an output file

```python
import numpy as np
import pandas as pd
import io

# Load
data    = np.load("data/processed/har/har_supervised.npz", allow_pickle=True)
signals = data["signals"]       # shape [N, 6, 100], dtype float32
meta    = pd.read_json(io.BytesIO(data["metadata"].item()))

print(signals.shape)            # (N, 6, 100)
print(meta.columns.tolist())    # ['sample_id', 'dataset_name', 'label_or_event', ...]
print(meta["label_or_event"].value_counts())  # activity label distribution
```

### Metadata schema (one row per sample)

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | str | Unique identifier, e.g. `pamap2_sup_sub3_rec_subject3.dat_w00042` |
| `dataset_name` | str | PAMAP2 / WISDM / EEGMMIDB / PTB-XL |
| `modality` | str | HAR / EEG / ECG |
| `subject_or_patient_id` | int/str | Enables subject-stratified splits |
| `source_file_or_record` | str | Original filename for provenance |
| `split` | str | pretrain / supervised / train / val / test |
| `label_or_event` | int or null | Unified class label (null for pretrain) |
| `sampling_rate_hz` | int | Sampling rate after preprocessing |
| `n_channels` | int | Number of signal channels |
| `n_samples` | int | Time samples per window |
| `channel_schema` | str | Comma-separated channel names |
| `qc_flags` | str | Pipe-separated QC notes (empty = clean) |

---

## Design decisions (summary)

See `reports/preprocessing_plan.md` for the full one-page design document.

| Decision | Choice | Why |
|----------|--------|-----|
| HAR target rate | 20 Hz | Match WISDM native rate; retains gesture dynamics ≤ 10 Hz |
| HAR channels | 6-ch acc+gyro at wrist | PAMAP2 wrist IMU ↔ WISDM watch; directly comparable |
| PAMAP2 class 0 | Excluded | Transient/null label — not a true activity |
| Pretrain windows | 10s, no overlap | Maximises unique segments; no label leakage |
| Supervised windows | 5s, 50% overlap | Data augmentation; majority-vote label assignment |
| EEG rate | 160 Hz (native) | Motor imagery energy ≤ 30 Hz; no resampling artefacts |
| ECG rate | 100 Hz (not 500 Hz) | 5× smaller; sufficient diagnostic bandwidth up to 40 Hz |
| ECG splits | folds 1–8 train, 9 val, 10 test | Officially recommended PTB-XL split; prevents patient leakage |

---

## Troubleshooting

### `bash: wget: command not found`
Install wget or let curl handle downloads automatically:
```bash
# macOS
brew install wget

# Ubuntu
sudo apt-get install wget

# Or just re-run — curl is the automatic fallback
bash setup_data.sh
```

### `python3: command not found` on Windows
Add Python to PATH during installation. Or use the full path:
```
C:\Python311\python.exe preprocess.py --config configs\pipeline_config.yaml
```

### `ModuleNotFoundError: No module named 'mne'`
Activate the virtual environment first:
```bash
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate.bat    # Windows
```

### PhysioNet download very slow or failing
PhysioNet applies rate limiting for large downloads.
The script retries automatically. You can also split the download:
```bash
# Download only runs 4 and re-run to pick up any missed files
bash setup_data.sh --skip-ecg   # skips PTB-XL, downloads EEG
```

### PTB-XL download fails with curl
Run the Python downloader directly:
```bash
python download_datasets.py --dataset ptbxl
```

### `unzip: command not found`
```bash
brew install unzip    # macOS
sudo apt install unzip  # Ubuntu
```
The script automatically falls back to Python's `zipfile` module if `unzip` is missing.

### Tests fail with `FileNotFoundError` for manifest/config
Tests check for processed files that may not exist yet. Run preprocessing first:
```bash
python preprocess.py --config configs/pipeline_config.yaml
```
Tests that depend on processed output auto-skip if the files are absent.

---

## Citation / data sources

| Dataset | Citation |
|---------|----------|
| PAMAP2 | Reiss, A. (2012). UCI ML Repository. https://doi.org/10.24432/C5NW2H |
| WISDM | Weiss, G. (2019). UCI ML Repository. https://doi.org/10.24432/C5HK59 |
| mHealth | Banos, O. et al. (2014). UCI ML Repository. https://doi.org/10.24432/C5TW22 |
| EEGMMIDB | Schalk et al. (2004). IEEE Trans Biomed Eng 51(6):1034-43. PhysioNet. |
| PTB-XL | Wagner et al. (2020). Nature Scientific Data. https://doi.org/10.13026/cmpy-7k07 |
