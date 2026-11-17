# Reliability of Physics-Guided Image Restoration under Spatially Varying Optical Aberrations

## Overview

This repository provides the implementation, optical aberration parameters, and numerical evaluation results associated with the manuscript submitted to the *Journal of Electronic Imaging*.

The study investigates the reliability of physics-guided image restoration under spatially varying optical aberrations, with particular attention to restoration accuracy, generalization, sensitivity to optical guidance, and robustness under synthetic noise.

## Repository Structure

- `code/`: Model training, optical degradation simulation, and evaluation scripts.
- `parameters/`: Optical aberration parameter files and validation-selected benchmark settings.
- `results/`: Numerical results for independent testing, optical guidance sensitivity, spatial analysis, and classical restoration comparisons.

## Experimental Configuration

The experiments use a physics-guided restoration model (AAPG) and an image-only baseline. The classical comparisons include locally approximated Wiener filtering and Richardson–Lucy deconvolution.

The classical benchmark parameters were selected using validation data and subsequently fixed for independent testing:

- Wiener regularization parameter: 0.0001
- Richardson–Lucy iterations: 20
- Spatial tiling: 3 × 3
- Overlap: 32 pixels
- Random seed: 2026

The final classical benchmark evaluates 200 independent test images under three synthetic noise conditions, with noise peak parameters of 0, 100, and 400.

## Reproducibility Notes

The Wiener and Richardson–Lucy implementations use local-constant PSF approximations rather than exact spatially varying deconvolution.

Noise experiments evaluate previously trained restoration models without retraining. The noise model is synthetic Poisson–Gaussian noise and should not be interpreted as measured sensor noise.

The repository contains evaluation results and configuration files. Model checkpoints and full image datasets are not included.

## Data and Code Availability

The repository provides source code, optical parameter files, and numerical results for inspection and reproduction. Re-execution of the complete experiments requires the corresponding image datasets and trained model checkpoints.

## Citation

Please cite the associated *Journal of Electronic Imaging* manuscript when using the code or experimental results.
