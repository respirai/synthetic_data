# HANDOVER – Synthetic ECG Data Generation for VivaLNK COPD using SimGAN

**Date:** Jan 11, 2026\
**Scope:** Handover notes / technical documentation for continuing development.

---

## 1) Context and Goal

**Goal:** Generate a larger synthetic dataset that mimics **VivaLNK chest-sensor ECG** recordings from **COPD** patients.

**Key requirement:** Generate **multi-beat ECG windows** where consecutive beats represent the **same patient** (stable morphology across beats).

**Baseline codebase (used/forked):**

- DreamStudioAI SimGAN implementation: [https://github.com/DreamStudioAI/sim\_gan/tree/master/sim\_gan](https://github.com/DreamStudioAI/sim_gan/tree/master/sim_gan)

**Reference method:**

- Golany et al. (ICML 2020), SimGANs: a GAN trained with an additional **Euler loss** derived from the McSharry ECG ODE simulator.

---

## 2) What the Original SimGAN Implementation Does

- Generates a **single heartbeat** (one cardiac cycle) per sample (**not** a continuous ECG strip).
- Uses MIT-BIH Arrhythmia data; beats extracted around R-peaks.
- Paper settings: **fs = 360 Hz**, beat length **L = 216 samples** (\~600 ms).
- Training strategy: typically trains a **separate GAN per heartbeat class** (Normal, VEB, SVEB, Fusion).

---

## 3) Target Domain: VivaLNK COPD – Key Differences

Target dataset: \~80 COPD patients measured with VivaLNK chest sensor.

Main differences to address:

- **Sampling frequency:** VivaLNK ECG is **128 Hz** (vs 360 Hz).
- **Scaling:** raw ECG uses a **magnification factor** (divide raw by magnification).
- **Output format:** need **multi-beat windows** that represent the same patient (morphology consistency).

---

## 4) Work Completed and Key Decisions

### 4.1 Baseline replication (done)

Re-ran training exactly as in the original repo/paper to validate:

- environment setup
- expected training behavior
- baseline output quality (single-beat)

### 4.2 Sampling-frequency adaptation (360 Hz → 128 Hz) (done)

Decision: keep internal SimGAN training at **360 Hz** and **resample VivaLNK to 360 Hz** at the I/O boundaries.

**Rationale**

- minimizes refactoring risk
- avoids simultaneously changing: generator output length + Euler/ODE integration step

### 4.3 Transfer learning / fine-tuning strategy (done)

Transfer learning is used: **MIT-BIH → VivaLNK COPD**.

Initialization decision:

- start from **SimDCGAN trained on Normal (N)** heartbeat class
- then fine-tune on VivaLNK COPD (closer “clean morphology” baseline)

Important note:

- COPD-specific simulator parameters **η** should be used (if/when implemented) during the **COPD fine-tuning** stage, not during MIT-BIH pretraining.

### 4.4 COPD-specific simulator parameter distribution p(η | COPD) (to reconsider / optional)

In the SimGAN paper, Euler loss samples simulator parameters **η \~ p(η | class)**.

Planned COPD adaptation:

- Replace with a COPD-specific prior: **η \~ p(η | COPD)**
- Practical plan:
  1. For real COPD beats (or short windows), estimate the best-fitting η by minimizing the simulator distance (same quantity used inside Euler loss)
  2. Collect η across COPD dataset
  3. Compute **μ\_COPD** and **Σ\_COPD** (often diagonal)
  4. During COPD fine-tuning, sample **η \~ N(μ\_COPD, Σ\_COPD)** inside the Euler loss

Status:

- This step was deferred after obtaining high-quality **single-beat** signals, to prioritize the multi-beat/window requirement first.

### 4.5 Multi-beat / full ECG generation requirement (in progress)

Current generator outputs a single beat. To generate “full ECG” or at least windows:

**Approach 1 (simple)**

- Increase generator output length to **T samples** and compute Euler loss over the entire window.

**Approach 2 (preferred)** Hierarchical model:

- beat morphology generator
- RR/HRV (rhythm) module
- window assembly module

**Same-patient constraint**

- keep a fixed **patient/style embedding** (or fixed η) for the entire window
- allow rhythm variability via RR/HRV (RR sequence)

---

## 5) Changes vs. Original SimGAN Repo (Short Summary)

**Upstream:** [https://github.com/DreamStudioAI/sim\_gan/tree/master](https://github.com/DreamStudioAI/sim_gan/tree/master)\
**This repo:** [https://github.com/respirai/synthetic\_data/tree/main](https://github.com/respirai/synthetic_data/tree/main)

Compared to the original SimGAN implementation, the following high-level changes were introduced to support **MIT-BIH → VivaLNK COPD transfer learning**:


Added VivaLNK/COPD dataset ingestion code (new loaders):
Added COPD-specific loader(s), notably data_reader/ecg_dataset_copd_from_csv.py.

Resampling support introduced for VivaLNK 128 Hz → internal 360 Hz pipeline:
Added data_reader/transforms_resample.py and documented the strategy: keep SimGAN internal pipeline at 360 Hz, resample at input/output boundaries to match VivaLNK 128 Hz.

Multi-beat / window-oriented simulator step added (prep for “ECG windows”):
Added dynamical_model/Euler/euler_window.py (upstream Euler module is single-step oriented). This is part of the path toward generating consecutive beats / windows rather than isolated beats.

Checkpoint-based generation utilities added:
Added gan_models/generate_from_checkpoint.py + gan_models/train_utils.py (not present in upstream). These support a workflow where you pretrain, fine-tune, then generate from a chosen checkpoint.

transfer learning plan written into repo:
 explicitly layout a two-stage pipeline: pretrain on MIT‑BIH → fine-tune on VivaLNK COPD, with placeholder flags like --PHASE pretrain/finetune and --CKPT. It also points to docs/HANDOVER.md for the detailed handover.
---

##

```bash
```

---

## 6) Expected Artifacts / Outputs

- MIT-BIH pretraining checkpoints\
  e.g., `checkpoints/simdcgan_N_*.pt`
- COPD η prior (if implemented)\
  e.g., `eta_copd.npz` containing (μ\_COPD, Σ\_COPD) + metadata
- COPD fine-tuned checkpoints\
  e.g., `checkpoints/copd_finetune_*.pt`
- Exported synthetic dataset\
  e.g., `synthetic/` containing ECG windows

---

## 7) Runbook (template)

> Replace script names and flags with your branch’s actual ones.

### Step 1 — Environment setup

```bash
# UNFORTUNATLEY THE requirements.txt FILE WAS IN THE PIPELINE_ORIGINAL MACHINE
# SO NEEDS TO INSTALL ALL PACKAGES 1 BY 1
#pip install -r requirements.txt 

```

### Step 2 — Pretrain on MIT-BIH (Normal class)

```bash
python train_sim_gan.py \
  --GAN_TYPE SimDCGAN \
  --BEAT_TYPE N \
  --BATCH_SIZE 64 \
  --NUM_ITERATIONS 20000 \
  --PHASE pretrain
```

### Step 3 — Fine-tune on VivaLNK COPD

```bash
python train_sim_gan.py \
  --GAN_TYPE SimDCGAN \
  --BEAT_TYPE N \
  --BATCH_SIZE 64 \
  --NUM_ITERATIONS 20000 \
  --PHASE finetune \
  --CKPT <path_to_pretrained_ckpt>
```

### Step 4 — Generate synthetic samples

```bash
python gan_models/generate_from_checkpoint.py --GAN_TYPE SimDCGAN --CKPT <path_to_finetuned_ckpt> --OUT_DIR synthetic/

```

---

## 8) Open Questions / Next Steps

- Decide canonical representation: beat-level vs window-level for downstream tasks.
- Finalize handling of \~76.8 samples/beat at 128 Hz (fixed L vs time-normalization).
- Implement and validate η-fitting on real COPD beats; sanity-check μ/σ ranges.
- Define evaluation metrics:
  - physiological plausibility
  - utility (TSTR: train-on-synthetic, test-on-real)
  - privacy (nearest-neighbor checks)
- Add **ACC generation** after ECG window generation is stable.

##

