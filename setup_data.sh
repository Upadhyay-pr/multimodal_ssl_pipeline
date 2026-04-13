#!/usr/bin/env bash
# =============================================================================
# setup_data.sh
# Shell wrapper: Python virtual environment, folder structure, dataset downloads.
#
# Supported platforms:
#   macOS  (Intel and Apple Silicon) — requires bash >= 3.2, curl (built-in)
#   Linux  (Ubuntu, Debian, CentOS) — wget preferred, curl fallback
#   Windows — run inside Git Bash, WSL (Ubuntu), or MSYS2
#
# Usage:
#   bash setup_data.sh [options]
#
# Options:
#   --skip-eeg     Skip EEGMMIDB download (~3 GB)
#   --skip-ecg     Skip PTB-XL download (~1.7 GB)
#   --no-mhealth   Skip bonus mHealth dataset
#   --skip-venv    Skip virtual environment creation
#   --dry-run      Print what would happen without downloading
#   --help         Show this message
#
# What this script does (and ONLY this):
#   1. Detect OS and available tools
#   2. Create Python virtual environment and install requirements
#   3. Create the project directory tree
#   4. Download raw data into the correct folders
#   5. Write a machine-readable JSON download manifest
#   6. Exit non-zero if any mandatory step fails
#
# All preprocessing, validation, and reporting are in Python scripts.
# =============================================================================

# ---------------------------------------------------------------------------
# Do NOT use strict set -e here; we handle errors explicitly per step.
# set -u catches undefined variables which is helpful.
# ---------------------------------------------------------------------------
set -uo pipefail

# =============================================================================
# COLOURS for friendly output (auto-disabled if not a terminal)
# =============================================================================
if [[ -t 1 ]]; then
  RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
  BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; YELLOW=''; GREEN=''; BLUE=''; BOLD=''; RESET=''
fi

info()  { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
fatal() { echo -e "${RED}[FATAL]${RESET} $*" >&2; exit 1; }
step()  { echo -e "\n${BOLD}${BLUE}=== $* ===${RESET}"; }

# =============================================================================
# OS DETECTION
# =============================================================================
OS_TYPE="$(uname -s 2>/dev/null || echo "unknown")"

case "$OS_TYPE" in
  Linux*)
    PLATFORM="linux"
    FILE_SIZE_CMD="stat -c%s"
    MD5_CMD="md5sum"
    ;;
  Darwin*)
    PLATFORM="macos"
    FILE_SIZE_CMD="stat -f%z"
    # macOS md5 output: "MD5 (file) = hash"  →  use -q to get just hash
    MD5_CMD="md5 -q"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM="windows_bash"
    FILE_SIZE_CMD="stat -c%s"
    MD5_CMD="md5sum"
    ;;
  *)
    PLATFORM="unknown"
    FILE_SIZE_CMD="stat -c%s"
    MD5_CMD="md5sum"
    warn "Unrecognised OS '${OS_TYPE}'. Defaulting to Linux-style commands."
    ;;
esac

info "Platform: ${OS_TYPE} (${PLATFORM})"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
SKIP_EEG=false
SKIP_ECG=false
DRY_RUN=false
INCLUDE_MHEALTH=true
SKIP_VENV=false

for arg in "$@"; do
  case $arg in
    --skip-eeg)    SKIP_EEG=true ;;
    --skip-ecg)    SKIP_ECG=true ;;
    --dry-run)     DRY_RUN=true ;;
    --no-mhealth)  INCLUDE_MHEALTH=false ;;
    --skip-venv)   SKIP_VENV=true ;;
    --help)
      echo ""
      echo "  Usage: bash setup_data.sh [options]"
      echo ""
      echo "  Options:"
      echo "    --skip-eeg     Skip EEGMMIDB download (~3 GB, uses --skip-eeg to save time)"
      echo "    --skip-ecg     Skip PTB-XL download (~1.7 GB)"
      echo "    --no-mhealth   Skip bonus mHealth dataset (~small)"
      echo "    --skip-venv    Skip Python virtual environment creation"
      echo "    --dry-run      Print actions without executing downloads"
      echo "    --help         Show this help"
      echo ""
      echo "  Platform notes:"
      echo "    macOS:   curl is used automatically (wget optional: brew install wget)"
      echo "    Linux:   wget preferred (sudo apt-get install wget)"
      echo "    Windows: run in Git Bash or WSL (Ubuntu recommended)"
      echo ""
      echo "  Quick start (skip large downloads for testing):"
      echo "    bash setup_data.sh --skip-eeg --skip-ecg --no-mhealth"
      echo ""
      exit 0 ;;
    *)
      error "Unknown argument: $arg"
      error "Run 'bash setup_data.sh --help' for usage."
      exit 1 ;;
  esac
done

# =============================================================================
# PATHS — anchored to the directory containing this script
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR" || fatal "Cannot cd to script directory: $SCRIPT_DIR"

RAW_DIR="data/raw"
LOG_DIR="logs"
MANIFEST_FILE="reports/download_manifest.json"
DOWNLOAD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="${LOG_DIR}/setup_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo 'session').log"

# =============================================================================
# DIRECTORY TREE — create everything upfront
# =============================================================================
step "Creating directory tree"

mkdir -p \
  "${RAW_DIR}/pamap2" \
  "${RAW_DIR}/wisdm" \
  "${RAW_DIR}/mhealth" \
  "${RAW_DIR}/eegmmidb" \
  "${RAW_DIR}/ptbxl" \
  data/interim/har \
  data/interim/eeg \
  data/interim/ecg \
  data/processed/har \
  data/processed/eeg \
  data/processed/ecg \
  reports \
  submission_sample/har \
  submission_sample/eeg \
  submission_sample/ecg \
  "${LOG_DIR}" || fatal "Failed to create directories."

info "Directory tree created."

# =============================================================================
# LOGGING — redirect to log file (compatible with bash 3.x on macOS)
# =============================================================================
# Tee approach: works on bash 3.2 (macOS default) through bash 5+
# We test process substitution first; fall back to plain file redirect.
_tee_ok=false
(exec > >(cat > /dev/null) 2>/dev/null) && _tee_ok=true || true

if [[ "$_tee_ok" == "true" ]]; then
  exec > >(tee -a "${LOG_FILE}") 2>&1
else
  # Bash 3.2 on some macOS versions: just write to log file only
  exec >> "${LOG_FILE}" 2>&1
  echo "[INFO] bash version does not support tee redirect; output goes to log only."
  echo "[INFO] Tail the log in another terminal: tail -f ${LOG_FILE}"
fi

info "Log file: ${LOG_FILE}"
info "Download date: ${DOWNLOAD_DATE}"

# =============================================================================
# DOWNLOAD TOOL DETECTION
# =============================================================================
step "Detecting download tools"

DOWNLOADER=""
if command -v wget &>/dev/null; then
  DOWNLOADER="wget"
  WGET_VERSION=$(wget --version 2>&1 | head -1 || echo "unknown")
  info "Download tool: wget  (${WGET_VERSION})"
elif command -v curl &>/dev/null; then
  DOWNLOADER="curl"
  CURL_VERSION=$(curl --version 2>&1 | head -1 || echo "unknown")
  info "Download tool: curl  (${CURL_VERSION})"
  warn "wget not found. curl will be used (slower for PhysioNet recursive downloads)."
  if [[ "$PLATFORM" == "macos" ]]; then
    warn "  → Install wget for faster PhysioNet downloads: brew install wget"
  elif [[ "$PLATFORM" == "linux" ]]; then
    warn "  → Install wget: sudo apt-get install wget"
  fi
else
  error "Neither wget nor curl found."
  error "  macOS:   brew install wget"
  error "  Ubuntu:  sudo apt-get install wget"
  error "  Windows: curl is bundled with Git for Windows"
  exit 1
fi

# =============================================================================
# PYTHON DETECTION AND VIRTUAL ENVIRONMENT SETUP
# =============================================================================
step "Setting up Python virtual environment"

PYTHON_CMD=""
for py in python3 python python3.12 python3.11 python3.10 python3.9; do
  if command -v "$py" &>/dev/null; then
    PY_VERSION=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 8 ]]; then
      PYTHON_CMD="$py"
      info "Found: $py  (version ${PY_VERSION})"
      break
    fi
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  error "Python 3.8+ is required but not found."
  error "  macOS:   brew install python"
  error "  Ubuntu:  sudo apt-get install python3"
  error "  Windows: download from https://python.org"
  exit 1
fi

VENV_DIR="venv"

if [[ "$SKIP_VENV" == "true" ]]; then
  warn "Skipping virtual environment setup (--skip-venv)."
elif [[ "$DRY_RUN" == "true" ]]; then
  info "[DRY-RUN] Would create venv at '${VENV_DIR}' and install requirements.txt"
else
  # Create venv if it doesn't already exist
  if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating virtual environment at '${VENV_DIR}'..."
    "$PYTHON_CMD" -m venv "${VENV_DIR}" || {
      warn "venv creation failed. Trying --without-pip fallback..."
      "$PYTHON_CMD" -m venv --without-pip "${VENV_DIR}" || {
        error "Cannot create venv. Install python3-venv:"
        error "  Ubuntu: sudo apt-get install python3-venv"
        error "  macOS:  brew install python"
        exit 1
      }
    }
    info "Virtual environment created."
  else
    info "Virtual environment '${VENV_DIR}' already exists."
  fi

  # Determine venv python / pip paths
  if [[ -f "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON="${VENV_DIR}/bin/python"
    VENV_PIP="${VENV_DIR}/bin/pip"
  elif [[ -f "${VENV_DIR}/Scripts/python.exe" ]]; then
    # Windows Git Bash / MSYS2
    VENV_PYTHON="${VENV_DIR}/Scripts/python"
    VENV_PIP="${VENV_DIR}/Scripts/pip"
  else
    fatal "Venv created but python binary not found at expected path. Check ${VENV_DIR}/"
  fi

  # Upgrade pip silently
  info "Upgrading pip..."
  "${VENV_PYTHON}" -m pip install --upgrade pip --quiet 2>&1 | tail -1 || true

  # Install requirements
  if [[ -f "requirements.txt" ]]; then
    info "Installing requirements.txt into venv..."
    "${VENV_PYTHON}" -m pip install -r requirements.txt --quiet 2>&1 || {
      error "pip install failed. Trying without version pins..."
      "${VENV_PYTHON}" -m pip install numpy pandas scipy pyyaml psutil mne wfdb matplotlib pytest || \
        error "Could not install required packages."
        error "Try manually:"
        error "  source ${VENV_DIR}/bin/activate"
        error "  pip install -r requirements.txt"
        exit 1
    }
    info "Requirements installed successfully."
  else
    warn "requirements.txt not found. Installing core packages..."
    "${VENV_PYTHON}" -m pip install numpy pandas scipy pyyaml psutil mne wfdb matplotlib pytest --quiet || true
  fi

  echo ""
  echo -e "${BOLD}${GREEN}Virtual environment ready.${RESET}"
  if [[ "$PLATFORM" == "windows_bash" ]]; then
    echo "  Activate with:  source ${VENV_DIR}/Scripts/activate"
  else
    echo "  Activate with:  source ${VENV_DIR}/bin/activate"
  fi
  echo ""
fi

# =============================================================================
# HELPERS: file size and MD5 (cross-platform)
# =============================================================================

get_file_size() {
  local path="$1"
  $FILE_SIZE_CMD "$path" 2>/dev/null || echo 0
}

get_md5() {
  local path="$1"
  if [[ -f "$path" ]]; then
    $MD5_CMD "$path" 2>/dev/null | awk '{print $1}' || echo ""
  else
    echo ""
  fi
}

# =============================================================================
# HELPER: download a single file (wget or curl)
# Retries 3 times with increasing delay. Fails clearly on mandatory downloads.
# =============================================================================
download_file() {
  local url="$1"
  local dest_dir="$2"
  local filename="$3"
  local dataset_name="$4"
  local mandatory="${5:-true}"
  local dest_path="${dest_dir}/${filename}"

  info "${dataset_name}: ${url}"

  if [[ "$DRY_RUN" == "true" ]]; then
    info "[DRY-RUN] Would download → ${dest_path}"
    return 0
  fi

  # Skip if already downloaded and non-trivial size
  if [[ -f "$dest_path" ]]; then
    local existing_size
    existing_size=$(get_file_size "$dest_path")
    if [[ "$existing_size" =~ ^[0-9]+$ ]] && [[ "$existing_size" -gt 10000 ]]; then
      info "${dataset_name}: already downloaded (${existing_size} bytes). Skipping."
      return 0
    else
      info "${dataset_name}: existing file is too small (${existing_size} bytes), re-downloading."
      rm -f "$dest_path"
    fi
  fi

  local dl_ok=false
  local attempt
  for attempt in 1 2 3; do
    info "${dataset_name}: download attempt ${attempt}/3..."
    if [[ "$DOWNLOADER" == "wget" ]]; then
      # --no-verbose instead of --quiet so we still see errors
      # --progress=dot:mega is widely supported across wget versions
      wget \
        --no-verbose \
        --tries=1 \
        --timeout=120 \
        --no-check-certificate \
        --output-document="${dest_path}" \
        "${url}" 2>&1 && dl_ok=true && break || true
    else
      curl \
        --location \
        --retry 1 \
        --max-time 300 \
        --fail \
        --output "${dest_path}" \
        "${url}" 2>&1 && dl_ok=true && break || true
    fi
    warn "${dataset_name}: attempt ${attempt} failed, retrying..."
    sleep $((attempt * 2))
  done

  if [[ -f "$dest_path" ]]; then
    local file_size
    file_size=$(get_file_size "$dest_path")
    if [[ "$file_size" =~ ^[0-9]+$ ]] && [[ "$file_size" -gt 10000 ]]; then
      info "${dataset_name}: downloaded successfully (${file_size} bytes)"
      return 0
    fi
  fi

  error "${dataset_name}: download FAILED from ${url}"
  rm -f "$dest_path" 2>/dev/null || true

  # Detect if this is a UCI repository outage (502/503 errors are server-side)
  echo ""
  echo -e "${BOLD}${YELLOW}--- Troubleshooting: ${dataset_name} download failed ---${RESET}"
  echo ""
  echo "  The UCI ML Repository (archive.ics.uci.edu) may be temporarily down."
  echo "  This is a known issue — the server occasionally returns 502/503 errors."
  echo ""
  echo "  OPTION 1: Download manually from Kaggle (works when UCI is down)"
  echo ""

  case "$dataset_name" in
    PAMAP2*)
      echo "    Primary: https://www.kaggle.com/datasets/nandinibagga/pamap2-dataset"
      echo "    Alt:     https://www.kaggle.com/datasets/jeetp22/pamap2"
      echo "    IMPORTANT: Download MUST contain Protocol/subject101.dat (raw sensor files)."
      echo "               If you see .npy files, that's preprocessed data — try the alt link."
      echo "    Steps: Click Download -> move zip to ${dest_dir}/ -> unzip"
      ;;
    WISDM*)
      echo "    Primary: https://www.kaggle.com/datasets/mashlyn/smartphone-and-smartwatch-activity-and-biometrics"
      echo "    Steps: Click Download -> move zip to ${dest_dir}/wisdm-dataset.zip -> unzip"
      ;;
    mHealth*|MHEALTH*)
      echo "    Primary: https://www.kaggle.com/datasets/gaurav2022/mobile-health"
      echo "    Alt:     https://www.kaggle.com/datasets/nirmalsankalana/mhealth-dataset-data-set"
      echo "    NOTE: Pipeline handles both .log files (original) and single .csv (Kaggle format)."
      echo "    Steps: Click Download -> move zip to ${dest_dir}/MHEALTHDATASET.zip -> unzip"
      ;;
    *)
      echo "    Search for '${dataset_name}' on https://www.kaggle.com/datasets"
      ;;
  esac

  echo ""
  echo "  OPTION 2: Wait for UCI to come back online, then re-run:"
  echo "    bash setup_data.sh"
  echo "    (Already-downloaded files are skipped automatically.)"
  echo ""
  echo "  OPTION 3: Use the Python fallback downloader:"
  local ds_lower
  ds_lower=$(echo "$dataset_name" | tr '[:upper:]' '[:lower:]')
  echo "    python download_datasets.py --dataset ${ds_lower}"
  echo ""
  echo -e "${BOLD}${YELLOW}-----------------------------------------------${RESET}"
  echo ""

  if [[ "$mandatory" == "true" ]]; then
    error "${dataset_name} is MANDATORY. Pipeline cannot proceed without it."
    error "After downloading manually, re-run: bash setup_data.sh"
    exit 1
  else
    warn "${dataset_name} is OPTIONAL (bonus). Continuing without it."
    return 1
  fi
}

# =============================================================================
# HELPER: unzip archive (cross-platform)
# =============================================================================
unzip_if_needed() {
  local zip_path="$1"
  local dest_dir="$2"
  local dataset_name="$3"

  if [[ ! -f "$zip_path" ]]; then
    warn "${dataset_name}: zip not found at ${zip_path}. Skipping extract."
    return 0
  fi

  # Check if already extracted (look for any non-zip files)
  local extracted_count
  extracted_count=$(find "$dest_dir" -not -name "*.zip" -type f 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$extracted_count" -gt 0 ]]; then
    info "${dataset_name}: already extracted (${extracted_count} files present)."
    return 0
  fi

  info "${dataset_name}: extracting ${zip_path} → ${dest_dir}"
  if [[ "$DRY_RUN" != "true" ]]; then
    if command -v unzip &>/dev/null; then
      unzip -q -o "$zip_path" -d "$dest_dir" 2>&1 || {
        warn "${dataset_name}: unzip returned non-zero. Inspect manually: unzip -t '${zip_path}'"
      }
    elif command -v python3 &>/dev/null; then
      info "  (using Python zipfile module as unzip fallback)"
      python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$zip_path" "$dest_dir" || warn "${dataset_name}: Python unzip failed."
    else
      warn "${dataset_name}: 'unzip' not found. Install it:\n  macOS: brew install unzip\n  Ubuntu: sudo apt-get install unzip"
    fi
  fi
}

# =============================================================================
# HELPER: PhysioNet recursive wget download (PTB-XL)
# Falls back to Python downloader if wget is unavailable.
# =============================================================================
physionet_wget() {
  local base_url="$1"
  local dest_dir="$2"
  local dataset_name="$3"
  local mandatory="${4:-true}"

  info "${dataset_name}: PhysioNet recursive download from ${base_url}"

  if [[ "$DRY_RUN" == "true" ]]; then
    info "[DRY-RUN] Would recursively download ${base_url} into ${dest_dir}"
    return 0
  fi

  # Skip if enough files already present
  local file_count
  file_count=$(find "$dest_dir" -type f 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$file_count" -gt 50 ]]; then
    info "${dataset_name}: ${file_count} files already present. Skipping."
    return 0
  fi

  if [[ "$DOWNLOADER" == "wget" ]]; then
    # --no-host-directories + --cut-dirs controls directory nesting
    # Avoid --show-progress (not in all wget versions)
    wget \
      --recursive \
      --no-parent \
      --no-check-certificate \
      --quiet \
      --tries=3 \
      --timeout=120 \
      --reject "index.html*,*.png,*.ico,*.css,*.js,*.htm" \
      --directory-prefix="${dest_dir}" \
      --no-host-directories \
      --cut-dirs=3 \
      "${base_url}" 2>&1 || {
        warn "${dataset_name}: wget recursive download encountered errors (network issues?)."
        warn "  Falling back to Python downloader..."
        _python_physionet_download "$base_url" "$dest_dir" "$dataset_name" "$mandatory"
        return $?
      }
  else
    # No wget → use Python downloader
    info "${dataset_name}: wget not available. Using Python downloader..."
    _python_physionet_download "$base_url" "$dest_dir" "$dataset_name" "$mandatory"
    return $?
  fi

  local file_count_after
  file_count_after=$(find "$dest_dir" -type f 2>/dev/null | wc -l | tr -d ' ')
  info "${dataset_name}: ${file_count_after} files downloaded."

  if [[ "$file_count_after" -lt 5 ]] && [[ "$mandatory" == "true" ]]; then
    error "${dataset_name}: too few files downloaded (${file_count_after})."
    error "Check connectivity to ${base_url}"
    error "Or run: python download_datasets.py --dataset ptbxl"
    exit 1
  fi
}

# Internal: call Python download helper for PhysioNet
_python_physionet_download() {
  local base_url="$1"
  local dest_dir="$2"
  local dataset_name="$3"
  local mandatory="${4:-true}"

  if [[ ! -f "download_datasets.py" ]]; then
    if [[ "$mandatory" == "true" ]]; then
      error "download_datasets.py not found and wget is unavailable. Cannot download ${dataset_name}."
      exit 1
    else
      warn "download_datasets.py not found; skipping ${dataset_name}."
      return 1
    fi
  fi

  local py="${PYTHON_CMD:-python3}"
  if [[ -f "${VENV_DIR:-venv}/bin/python" ]]; then
    py="${VENV_DIR:-venv}/bin/python"
  fi

  local ds_lower
  ds_lower=$(echo "$dataset_name" | tr '[:upper:]' '[:lower:]')

  "$py" download_datasets.py --dataset "${ds_lower}" --dest-dir "${dest_dir}" 2>&1 || {
    if [[ "$mandatory" == "true" ]]; then
      error "Python download failed for ${dataset_name}."
      error "Try manually: python download_datasets.py --dataset ${ds_lower}"
      exit 1
    else
      warn "Python download failed for ${dataset_name}; continuing."
      return 1
    fi
  }
}

# =============================================================================
# HELPER: EEGMMIDB — download runs 4, 8, 12 only for all 109 subjects
# Uses individual file downloads (works with both wget and curl)
# =============================================================================
physionet_wget_eeg() {
  local dest_dir="$1"
  local mandatory="${2:-true}"
  local base_url="https://physionet.org/files/eegmmidb/1.0.0"

  info "EEGMMIDB: downloading motor imagery runs 4, 8, 12 for all 109 subjects"
  info "EEGMMIDB: target directory: ${dest_dir}"

  if [[ "$DRY_RUN" == "true" ]]; then
    info "[DRY-RUN] Would download 109 × 3 = 327 EDF files into ${dest_dir}"
    return 0
  fi

  # Skip if enough EDF files already present
  local edf_count
  edf_count=$(find "$dest_dir" -name "*.edf" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$edf_count" -gt 50 ]]; then
    info "EEGMMIDB: ${edf_count} EDF files already present. Skipping."
    return 0
  fi

  local failed=0
  local downloaded=0
  local subj

  # Loop over all 109 subjects, downloading 3 runs each
  for subj in $(seq 1 109); do
    local subj_id
    subj_id="$(printf 'S%03d' "$subj")"
    local subj_dir="${dest_dir}/${subj_id}"
    mkdir -p "$subj_dir"

    for run in 04 08 12; do
      local edf_file="${subj_id}R${run}.edf"
      local url="${base_url}/${subj_id}/${edf_file}"
      local out_path="${subj_dir}/${edf_file}"

      if [[ -f "$out_path" ]] && [[ "$(get_file_size "$out_path")" -gt 1000 ]]; then
        continue   # already downloaded
      fi

      local dl_success=false
      if [[ "$DOWNLOADER" == "wget" ]]; then
        wget --quiet --tries=2 --timeout=60 --no-check-certificate \
          -O "$out_path" "$url" 2>/dev/null && dl_success=true || true
      else
        curl --silent --location --retry 2 --max-time 60 --fail \
          --output "$out_path" "$url" 2>/dev/null && dl_success=true || true
      fi

      if [[ "$dl_success" == "true" ]] && [[ -f "$out_path" ]]; then
        local sz
        sz=$(get_file_size "$out_path")
        if [[ "$sz" =~ ^[0-9]+$ ]] && [[ "$sz" -gt 1000 ]]; then
          downloaded=$((downloaded + 1))
        else
          rm -f "$out_path"
          failed=$((failed + 1))
        fi
      else
        rm -f "$out_path" 2>/dev/null || true
        failed=$((failed + 1))
      fi
    done
  done

  local total_edf
  total_edf=$(find "$dest_dir" -name "*.edf" | wc -l | tr -d ' ')
  info "EEGMMIDB: ${total_edf} EDF files downloaded (${failed} failures)."

  if [[ "$total_edf" -lt 10 ]] && [[ "$mandatory" == "true" ]]; then
    error "Too few EEG files downloaded (${total_edf} < 10)."
    error "This usually means PhysioNet is rate-limiting connections."
    error "Solutions:"
    error "  1. Re-run this script (it resumes automatically)"
    error "  2. Check connectivity: curl -I https://physionet.org"
    error "  3. Use: python download_datasets.py --dataset eegmmidb"
    exit 1
  elif [[ "$failed" -gt 50 ]]; then
    warn "EEGMMIDB: ${failed} files failed to download. Re-run to retry missing files."
  fi
}

# =============================================================================
# MANIFEST accumulator
# =============================================================================
MANIFEST_ENTRIES=""

add_manifest_entry() {
  local dataset="$1"
  local url="$2"
  local local_path="$3"
  local status="$4"
  local note="${5:-}"

  local file_size="0"
  local file_hash=""

  if [[ -f "$local_path" ]]; then
    file_size=$(get_file_size "$local_path")
    file_hash=$(get_md5 "$local_path")
  elif [[ -d "$local_path" ]]; then
    local fc
    fc=$(find "$local_path" -type f 2>/dev/null | wc -l | tr -d ' ')
    file_size="dir:${fc} files"
  fi

  # Escape double quotes in note field
  note="${note//\"/\'}"

  local entry
  entry=$(cat <<JSONEOF
    {
      "dataset": "${dataset}",
      "url": "${url}",
      "local_path": "${local_path}",
      "download_date": "${DOWNLOAD_DATE}",
      "status": "${status}",
      "size_bytes": "${file_size}",
      "md5": "${file_hash}",
      "note": "${note}"
    }
JSONEOF
  )

  if [[ -n "$MANIFEST_ENTRIES" ]]; then
    MANIFEST_ENTRIES="${MANIFEST_ENTRIES},${entry}"
  else
    MANIFEST_ENTRIES="${entry}"
  fi
}

# =============================================================================
# DOWNLOADS
# =============================================================================

step "Downloading HAR datasets"

# ---------------------------------------------------------------------------
# PAMAP2 (mandatory)
# ---------------------------------------------------------------------------
PAMAP2_URL="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
PAMAP2_ZIP="${RAW_DIR}/pamap2/PAMAP2_Dataset.zip"

# Try primary URL, fall back to legacy UCI URL
download_file "$PAMAP2_URL" "${RAW_DIR}/pamap2" "PAMAP2_Dataset.zip" "PAMAP2" "true" || {
  warn "PAMAP2: primary URL failed. Trying legacy URL..."
  PAMAP2_URL_LEGACY="https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
  download_file "$PAMAP2_URL_LEGACY" "${RAW_DIR}/pamap2" "PAMAP2_Dataset.zip" "PAMAP2" "true"
  PAMAP2_URL="$PAMAP2_URL_LEGACY"
}

if [[ "$DRY_RUN" != "true" ]] && [[ -f "$PAMAP2_ZIP" ]]; then
  unzip_if_needed "$PAMAP2_ZIP" "${RAW_DIR}/pamap2" "PAMAP2"
fi
add_manifest_entry "PAMAP2" "$PAMAP2_URL" "${RAW_DIR}/pamap2" "downloaded" \
  "100 Hz wrist+chest+ankle IMU; 9 subjects; 18 activities"

# ---------------------------------------------------------------------------
# WISDM (mandatory)
# ---------------------------------------------------------------------------
WISDM_URL="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
WISDM_ZIP="${RAW_DIR}/wisdm/wisdm-dataset.zip"

download_file "$WISDM_URL" "${RAW_DIR}/wisdm" "wisdm-dataset.zip" "WISDM" "true" || {
  warn "WISDM: primary URL failed. Trying legacy URL..."
  WISDM_URL_LEGACY="https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
  download_file "$WISDM_URL_LEGACY" "${RAW_DIR}/wisdm" "wisdm-dataset.zip" "WISDM" "true"
  WISDM_URL="$WISDM_URL_LEGACY"
}

if [[ "$DRY_RUN" != "true" ]] && [[ -f "$WISDM_ZIP" ]]; then
  unzip_if_needed "$WISDM_ZIP" "${RAW_DIR}/wisdm" "WISDM"
fi
add_manifest_entry "WISDM" "$WISDM_URL" "${RAW_DIR}/wisdm" "downloaded" \
  "20 Hz phone+watch acc+gyr; 51 subjects; 18 activities"

# ---------------------------------------------------------------------------
# mHealth (optional bonus)
# ---------------------------------------------------------------------------
if [[ "$INCLUDE_MHEALTH" == "true" ]]; then
  step "Downloading bonus HAR/ECG dataset: mHealth"
  MHEALTH_URL="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
  MHEALTH_ZIP="${RAW_DIR}/mhealth/MHEALTHDATASET.zip"

  download_file "$MHEALTH_URL" "${RAW_DIR}/mhealth" "MHEALTHDATASET.zip" "mHealth" "false" || {
    MHEALTH_URL_LEGACY="https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
    download_file "$MHEALTH_URL_LEGACY" "${RAW_DIR}/mhealth" "MHEALTHDATASET.zip" "mHealth" "false" || true
    MHEALTH_URL="${MHEALTH_URL_LEGACY}"
  }

  if [[ "$DRY_RUN" != "true" ]] && [[ -f "$MHEALTH_ZIP" ]]; then
    unzip_if_needed "$MHEALTH_ZIP" "${RAW_DIR}/mhealth" "mHealth"
  fi
  add_manifest_entry "mHealth" "$MHEALTH_URL" "${RAW_DIR}/mhealth" "downloaded" \
    "Bonus: 50 Hz; chest+wrist+ankle; 2-lead ECG; 10 subjects"
else
  info "mHealth: skipped (--no-mhealth)"
  add_manifest_entry "mHealth" \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip" \
    "${RAW_DIR}/mhealth" "skipped" "Skipped by --no-mhealth flag"
fi

# ---------------------------------------------------------------------------
# EEGMMIDB (mandatory unless --skip-eeg)
# ---------------------------------------------------------------------------
if [[ "$SKIP_EEG" == "false" ]]; then
  step "Downloading EEG: EEGMMIDB (runs 4, 8, 12 only — ~3 GB total)"
  info "  This downloads 109 subjects × 3 runs = 327 EDF files."
  info "  Add --skip-eeg to bypass if bandwidth is limited."
  physionet_wget_eeg "${RAW_DIR}/eegmmidb" "true"
  add_manifest_entry "EEGMMIDB" \
    "https://physionet.org/files/eegmmidb/1.0.0/" \
    "${RAW_DIR}/eegmmidb" "downloaded" \
    "160 Hz EDF+; 64-ch; 109 subjects; runs 4,8,12 only (~3 GB)"
else
  info "EEGMMIDB: skipped (--skip-eeg)"
  add_manifest_entry "EEGMMIDB" \
    "https://physionet.org/files/eegmmidb/1.0.0/" \
    "${RAW_DIR}/eegmmidb" "skipped" "Skipped by --skip-eeg flag"
fi

# ---------------------------------------------------------------------------
# PTB-XL (mandatory unless --skip-ecg)
# ---------------------------------------------------------------------------
if [[ "$SKIP_ECG" == "false" ]]; then
  step "Downloading ECG: PTB-XL (~1.7 GB)"
  info "  Add --skip-ecg to bypass if bandwidth is limited."
  physionet_wget \
    "https://physionet.org/files/ptb-xl/1.0.3/" \
    "${RAW_DIR}/ptbxl" \
    "PTB-XL" \
    "true"
  add_manifest_entry "PTB-XL" \
    "https://physionet.org/files/ptb-xl/1.0.3/" \
    "${RAW_DIR}/ptbxl" "downloaded" \
    "12-lead ECG; 21801 records; 18869 patients; 100/500 Hz"
else
  info "PTB-XL: skipped (--skip-ecg)"
  add_manifest_entry "PTB-XL" \
    "https://physionet.org/files/ptb-xl/1.0.3/" \
    "${RAW_DIR}/ptbxl" "skipped" "Skipped by --skip-ecg flag"
fi

# =============================================================================
# WRITE DOWNLOAD MANIFEST
# =============================================================================
step "Writing download manifest"

cat > "$MANIFEST_FILE" <<MANIFESTEOF
{
  "manifest_type": "download_manifest",
  "generated_at": "${DOWNLOAD_DATE}",
  "pipeline_version": "1.0.0",
  "os": "${OS_TYPE}",
  "platform": "${PLATFORM}",
  "downloader": "${DOWNLOADER}",
  "entries": [
${MANIFEST_ENTRIES}
  ]
}
MANIFESTEOF

info "Manifest written to ${MANIFEST_FILE}"

# =============================================================================
# SUMMARY
# =============================================================================
step "Setup complete"

echo ""
echo -e "${BOLD}Next steps:${RESET}"
echo ""

if [[ "$SKIP_VENV" == "false" ]] && [[ "$DRY_RUN" == "false" ]]; then
  if [[ "$PLATFORM" == "windows_bash" ]]; then
    echo "  1. Activate venv:    source ${VENV_DIR}/Scripts/activate"
  else
    echo "  1. Activate venv:    source ${VENV_DIR}/bin/activate"
  fi
fi

echo "  2. Preprocess all:   python preprocess.py --config configs/pipeline_config.yaml"
echo "  3. Validate outputs: python validate_outputs.py --config configs/pipeline_config.yaml"
echo "  4. Run tests:        python -m pytest tests/ -v"
echo ""
echo "  Modality-specific preprocessing:"
echo "    python preprocess.py --modality har"
echo "    python preprocess.py --modality eeg"
echo "    python preprocess.py --modality ecg"
echo ""
echo "  Log saved to: ${LOG_FILE}"
echo ""
