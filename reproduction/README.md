# InterPLM Reproduction

This directory documents my reproduction of selected experiments from InterPLM, with the goal of understanding sparse autoencoder-based interpretability for protein language models.

## Environment

Platform: macOS, Apple Silicon (arm64)

The original `env.yml` could not be resolved on `osx-arm64` because the pinned Conda packages `pytorch::pytorch==2.8.0` and `pytorch::torchvision==0.21.0` were unavailable for this platform.

I therefore created a Python 3.11 Conda environment and installed an ARM-compatible PyTorch build via pip.

- Python: 3.11
- PyTorch: 2.14.1
- MPS available: True

## Reproduction Progress

### Level 1 — Pretrained SAE
- [ ] Load ESM-2
- [ ] Extract hidden representations
- [ ] Load a pretrained InterPLM SAE
- [ ] Inspect sparse feature activations

### Level 2 — SAE Training
- [ ] Prepare protein representations
- [ ] Train an SAE
- [ ] Evaluate reconstruction and sparsity
- [ ] Inspect learned features