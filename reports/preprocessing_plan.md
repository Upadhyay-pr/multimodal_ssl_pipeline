# Preprocessing Plan — Multimodal SSL Pipeline
**Candidate:** Pragati Upadhyay | **Role:** Research Assistant MED05664 | **Date:** April 2026

---

## 1 · Channel Schema and Sensor Pairing

| Dataset | Selected Stream | Channels (6-ch schema) | Justification |
|---------|----------------|------------------------|---------------|
| PAMAP2 | Wrist (hand) IMU | acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z | Wrist is the closest analogue to a smartwatch; provides both acc and gyro at 100 Hz |
| WISDM | Watch sensor | acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z | Dominant-hand watch is directly comparable to PAMAP2 wrist IMU |
| mHealth (bonus) | Right wrist | acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z | Only wrist position with both acc and gyro available |
| EEGMMIDB | All 64 ch (10-10 system) | 64 EEG channels | Full-channel retention preserves spatial structure needed for motor imagery classification |
| PTB-XL | All 12 leads | I, II, III, aVR, aVL, aVF, V1–V6 | Standard clinical 12-lead; dropping any lead would limit diagnostic coverage |

---

## 2 · Sampling Rate Choices

| Dataset | Source Hz | Target Hz | Method | Reason |
|---------|-----------|-----------|--------|--------|
| PAMAP2 | 100 | 20 | `scipy.signal.resample_poly` (polyphase anti-alias) | Match WISDM; retain gesture dynamics up to 10 Hz |
| WISDM | 20 | 20 | None | Already at target |
| mHealth | 50 | 20 | `resample_poly` | Match HAR target |
| EEGMMIDB | 160 | 160 | None (kept native) | Motor imagery energy is 8–30 Hz; 160 Hz gives comfortable margin; resampling EEG introduces interpolation artefacts |
| PTB-XL | 100 or 500 | **100** | Native (no resampling) | 100 Hz captures all clinically relevant ECG morphology up to 40 Hz; 5× smaller file size vs 500 Hz; sufficient for standard diagnostic tasks |

---

## 3 · Window Definitions and Split Strategy

### HAR
| Mode | Window | Step | Labels | Purpose |
|------|--------|------|--------|---------|
| Pretrain | 10 s (200 samples) | 200 (no overlap) | None | Contiguous unlabelled segments for SSL masking/contrastive pre-training |
| Supervised | 5 s (100 samples) | 50 (50% overlap) | Majority vote | Fine-tuning; overlap provides data augmentation and smoother evaluation |

**Label assignment algorithm:** for each supervised window, the label is the *majority vote* over the resampled label array within that window's time span. This is conservative — mixed-activity windows where no single class exceeds 50% of frames are retained with the plurality label and flagged with `mixed_activity` in `qc_flags`. Windows coinciding entirely with PAMAP2 class 0 (transient) are **dropped**.

### EEG
- Fixed 4-second windows starting at **T1 or T2 event onset** (left/right fist imagery, runs 4, 8, 12).
- T0 (rest) excluded by default; can be re-enabled via config.
- No overlap: each event is one sample to prevent temporal data leakage between events.

### ECG (PTB-XL)
- Records are already 10 s; no windowing needed — one record = one sample.
- Patient-level splits use `strat_fold`: folds 1–8 → train, fold 9 → val, fold 10 → test. This is the officially recommended PTB-XL split to prevent patient-level leakage.

---

## 4 · Missing Values, Invalid Rows, and Null Labels

| Issue | Strategy |
|-------|----------|
| Short NaN gaps ≤ 10 samples (~0.5 s at 20 Hz) | Linear interpolation |
| Longer gaps | `ffill` + `bfill`; gap length logged in `qc_flags` |
| PAMAP2 class 0 (transient) | **Excluded entirely** — represents sensor artefact / transition periods |
| WISDM activities with no PAMAP2 equivalent | Mapped to closest activity or excluded (fully documented in config) |
| mHealth label 0 (null) | **Excluded** — same policy as PAMAP2 class 0 |
| Flat EEG channels (range < 1 µV) | NaN-filled; count reported in `qc_flags` |
| EEG amplitude > 500 µV | Treated as artefact; channel NaN-filled |
| ECG NaN / Inf after bandpass | `np.nan_to_num(0.0)`; flagged in `qc_flags` |

---

## 5 · Preprocessing Steps Summary

**HAR:** median filter (not applied — polyphase resampling provides implicit anti-aliasing) → linear interpolation → resample to 20 Hz → per-channel z-score (within subject).

**EEG:** load EDF+ with MNE → resample to 160 Hz if needed → common average reference → Butterworth bandpass 1–40 Hz (FIR) → 60 Hz notch → artefact channel detection → event-aligned windowing → per-channel z-score.

**ECG:** load with WFDB → Butterworth bandpass 0.5–40 Hz → per-record z-score.

---

## 6 · Expected Storage and Peak RAM

| Modality | Raw download | Processed output | Peak RAM |
|----------|-------------|-----------------|---------|
| HAR (all) | ~500 MB | ~240 MB | ~250 MB |
| EEG (runs 4,8,12 only) | ~3 GB | ~500 MB | ~500 MB |
| ECG (PTB-XL 100 Hz) | ~1.7 GB | ~1 GB | ~1 GB |
| **Total** | **~5.2 GB** | **~1.7 GB** | **~1.5 GB** |

**Main engineering risks:**
1. **PhysioNet rate limiting** — EEGMMIDB has 109 subjects × 3 runs = 327 files; wget with `--tries=3` and `--timeout=120` handles transient failures.
2. **PAMAP2 irregular sampling** — Columns occasionally have NaN runs > 1 s due to sensor dropout; handled by gap-length threshold and ffill fallback.
3. **EEG memory** — Loading all 109 subjects × 3 runs simultaneously would require ~5 GB RAM; the pipeline processes subject-by-run in a streaming loop.
4. **ECG IO bottleneck** — WFDB reads 21 k records sequentially; this is the longest step (~20–30 min). Parallelisation with `multiprocessing.Pool` is straightforward if needed.
