# Synthetic ECG Data Generation for VivaLNK COPD (SimGAN-based)

This project adapts **SimGAN** (Simulator-Based GAN with an additional **Euler loss** derived from the McSharry ECG ODE simulator) to generate **synthetic ECG** that mimics **VivaLNK chest sensor** recordings from **COPD** patients.

- Upstream codebase used: `DreamStudioAI/sim_gan`  
  https://github.com/DreamStudioAI/sim_gan/tree/master/sim_gan
- Reference paper: Golany et al., *SimGANs: Simulator-Based Generative Adversarial Networks for ECG Synthesis…* (ICML 2020)

## What we are trying to produce
- **Multi-beat ECG windows** (continuous segments) where consecutive beats represent the **same patient** (stable morphology).
- Eventually: **ECG** aligned to VivaLNK device characteristics.

## Current status (high level)
- ✅ Baseline SimGAN training reproduced on MIT-BIH (single-beat generation).
- ✅ Transfer learning plan established: **MIT-BIH → VivaLNK COPD**.
- ✅ Sampling-frequency strategy chosen: keep internal pipeline at **360 Hz** and resample VivaLNK (**128 Hz**) at I/O.
- 🚧 Multi-beat / full-window generation is **in progress**.
- ⚠️ COPD-specific simulator-parameter prior **p(η | COPD)** planned but deferred until window generation is stable.

## Quick start (template)
> Update script names/flags to match your branch.

```bash
# 1) install deps
# UNFORTUNATLEY THE requirements.txt FILE WAS IN THE PIPELINE_ORIGINAL MACHINE
# SO NEEDS TO INSTALL ALL PACKAGES 1 BY 1
#pip install -r requirements.txt 

# 2) pretrain on MIT-BIH (Normal class)
python gan_models/train_sim_gan.py --GAN_TYPE SimDCGAN --BEAT_TYPE N --PHASE pretrain --BATCH_SIZE 64 --NUM_ITERATIONS 20000 

# 3) fine-tune on VivaLNK COPD
python gan_models/train_sim_gan.py --GAN_TYPE SimDCGAN --BEAT_TYPE N --PHASE finetune --CKPT <path_to_pretrained_ckpt> --BATCH_SIZE 64 --NUM_ITERATIONS 20000 

# 4) generate synthetic samples
python gan_models/generate_from_checkpoint.py --GAN_TYPE SimDCGAN --CKPT <path_to_finetuned_ckpt> --OUT_DIR synthetic/
```

## Data notes (VivaLNK)
- Target ECG sampling frequency: **128 Hz**.
- Raw ECG values require scaling using a **magnification factor** (device specification).
- Current approach keeps SimGAN internals at 360 Hz and performs resampling at the loader/export boundaries.

## Documentation
- Full handover / technical notes: **[`docs/HANDOVER.md`](docs/HANDOVER.md)**
