# Downstream Self-Supervised Learning — Integration Note

**Assessment:** MED05664 | **Date:** April 2026

---

## How the Processed Outputs Feed a Downstream SSL Pipeline

The four output files produced by this pipeline (`har_pretrain.npz`, `har_supervised.npz`,
`eeg_motor_imagery.npz`, `ecg_ptbxl.npz`) share a common shape convention — `[N, C, T]`
(batch × channels × time samples) — and are stored as `float32` arrays that can be loaded
directly into any deep-learning framework without further conversion.

---

## 1 · Pre-training Phase (Self-Supervised)

The goal of SSL is to learn useful representations from unlabelled data by solving a
pretext task that does not require manual annotation.

### Inputs used

| File | Shape | Use |
|------|-------|-----|
| `har_pretrain.npz` | [N, 6, 200] | 10s IMU segments, no labels |
| `eeg_motor_imagery.npz` | [N, 64, 640] | 4s EEG windows (T1/T2) |
| `ecg_ptbxl.npz` | [N, 12, 1000] | 10s 12-lead ECG records |

### Three compatible SSL approaches

#### A — Masked Autoencoder (MAE / masked patch modelling)

```
Input:  signal [B, C, T]  (e.g. [32, 6, 200] for HAR)
Step 1: Divide T into P non-overlapping patches of length p
        → tokens: [B, C, P]  (e.g. p=10 gives P=20 patches)
Step 2: Randomly mask ~75% of patches (set to learnable mask token)
Step 3: Encoder (e.g. Transformer) processes visible patches only
Step 4: Decoder reconstructs ALL P patches from encoded context
Loss:   MSE between predicted and true masked patches
Goal:   Encoder learns to capture global temporal structure
```

**Why it suits this data:** HAR and EEG signals are highly correlated across time.
Masking patches forces the model to learn the relationship between distant segments
(e.g. arm acceleration implies the next motion phase).

#### B — Contrastive Learning (SimCLR / BYOL style)

```
Input:  signal [B, C, T]
Step 1: Apply two stochastic augmentations to each sample:
        aug_1 ← random crop + Gaussian noise + channel dropout
        aug_2 ← random crop + time warp + scaling
Step 2: Encoder f(.) maps both views to embeddings z1, z2
Step 3: Projection head g(.) maps to lower-dim representations
Loss:   NT-Xent (SimCLR): maximise similarity(z1, z2) and
        push apart different-sample pairs in the batch
Goal:   Invariant representation under physiological noise
```

**Why it suits this data:** Intra-subject variability in IMU and ECG (scaling,
speed, electrode placement drift) is the natural source of augmentation.
The same activity recorded at different intensities should map to the same representation.

#### C — Temporal Neighbourhood Coding (TNC / CPC)

```
Input:  long continuous signal segment [C, T_long]
Step 1: Sample a reference window w_ref at position t
Step 2: Sample a positive window w_pos close in time (|Δt| ≤ δ)
Step 3: Sample a negative window w_neg far in time (|Δt| > Δ)
Loss:   Binary cross-entropy: does the encoder correctly
        identify whether a candidate window is a temporal
        neighbour of the reference?
Goal:   Representations that are smooth over time
```

**Why it suits this data:** Physiological signals evolve continuously — consecutive
EEG or HAR windows should have similar representations. TNC directly encodes
this temporal smoothness assumption.

---

## 2 · Fine-tuning Phase (Supervised)

Once pre-training is complete, the encoder is frozen (or lightly fine-tuned) and
a shallow classification head is attached.

### Inputs used

| File | Shape | Labels | Task |
|------|-------|--------|------|
| `har_supervised.npz` | [N, 6, 100] | 7 activity classes | Activity recognition |
| `eeg_motor_imagery.npz` | [N, 64, 640] | T1/T2 (left/right fist) | Motor imagery BCI |
| `ecg_ptbxl.npz` | [N, 12, 1000] | NORM/MI/STTC/CD/HYP | ECG diagnosis |

```
Pre-trained encoder (frozen weights)
           ↓
   Global average pool or [CLS] token
           ↓
   Linear layer (or 2-layer MLP)
           ↓
   Softmax → class probabilities
Loss: cross-entropy on labelled examples only
```

### Key benefit: leakage-free evaluation

Every processed window carries a `subject_or_patient_id` field.
This means downstream splits can be constructed at the *subject level*:

```python
# Example: subject-stratified split (pseudocode)
import numpy as np, json, io

data   = np.load("data/processed/har/har_supervised.npz", allow_pickle=True)
meta   = pd.read_json(io.BytesIO(data["metadata"].item()))
signals = data["signals"]          # [N, 6, 100]  float32

# Group by subject, split 70/15/15
subjects = meta["subject_or_patient_id"].unique()
np.random.shuffle(subjects)
n = len(subjects)
train_subj = subjects[:int(0.7*n)]
val_subj   = subjects[int(0.7*n):int(0.85*n)]
test_subj  = subjects[int(0.85*n):]

train_mask = meta["subject_or_patient_id"].isin(train_subj)
# → No subject appears in both train and test → no leakage
```

For PTB-XL, the `strat_fold` column already encodes the officially recommended
patient-level split (folds 1–8 train, 9 val, 10 test).

---

## 3 · Data Loader Pseudocode (PyTorch-style)

```python
# PyTorch familiarity is not required — this is pseudocode to illustrate the concept

class MultimodalSSLDataset:
    """
    Generic dataset wrapper for .npz files produced by this pipeline.
    Works for all three modalities since they share [N, C, T] convention.
    """
    def __init__(self, npz_path, split="train", transform=None):
        data = np.load(npz_path, allow_pickle=True)
        self.signals  = data["signals"]         # [N, C, T]  float32
        self.metadata = pd.read_json(...)        # N rows

        # Filter to requested split
        mask = self.metadata["split"] == split
        self.signals  = self.signals[mask]
        self.metadata = self.metadata[mask]
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]                  # [C, T]
        label = self.metadata.iloc[idx]["label_or_event"]
        if self.transform:
            x = self.transform(x)
        return x, label                        # tensor, int

# Pre-training: no labels, apply two augmentations
class ContrastiveDataset(MultimodalSSLDataset):
    def __getitem__(self, idx):
        x = self.signals[idx]
        return augment(x), augment(x)          # two views, no label needed

# DataLoader
train_dataset = MultimodalSSLDataset("data/processed/har/har_pretrain.npz")
# batch = [B, C, T]  → ready for any encoder (1-D CNN, Transformer, etc.)
```

---

## 4 · Why This Preprocessing Supports SSL Specifically

| Design decision | Benefit for SSL |
|-----------------|-----------------|
| Float32, no handcrafted features | Raw signals preserve all information the model can learn from |
| Per-channel z-score normalisation | Removes inter-subject scale differences that would bias contrastive loss |
| 10s pretrain windows (HAR) | Long enough for temporal context; short enough to fit in GPU memory |
| No overlap in pretrain set | Avoids temporal leakage between training samples; each window is independent |
| Subject ID preserved | Enables subject-stratified splits to prevent identifiability leakage |
| `[N, C, T]` convention | Compatible with all major time-series encoders (e.g. TS-TCC, BIOT, Mamba-EEG) |
| 100 Hz ECG (not 500 Hz) | 5× smaller memory footprint; 5× faster batch loading; no model accuracy loss for standard diagnostic bandwidth |

---

## 5 · Practical Compute Considerations

| Stage | Bottleneck | Recommendation |
|-------|-----------|----------------|
| SSL pre-training (HAR) | GPU memory | Batch size 256 on a single 8 GB GPU is feasible for [256, 6, 200] |
| SSL pre-training (EEG) | GPU memory | [256, 64, 640] ≈ 100 MB/batch; reduce batch size to 64 on smaller GPUs |
| SSL pre-training (ECG) | IO throughput | Pre-load all 21k records into RAM (~1 GB); or use memory-mapped numpy |
| Fine-tuning | Labelled data size | HAR: 9 subjects × few hundred windows → easy to overfit; use heavy dropout |
| Multi-modal fusion | Alignment | Timestamp alignment was not enforced across modalities; ensure downstream model handles asynchronous inputs |

---

*This note is part of the MED05664 candidate submission.
The pipeline code is in `preprocess.py`; validation in `validate_outputs.py`.*
