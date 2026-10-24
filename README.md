# Spectrum autoencoder

Fully connected autoencoder for binned transmission spectra (POS dataset).

## Data (`data/`)

| File | Shape | |
|---|---|---|
| `pos_spec_binned.npy` | (98725, 52) | binned spectra (transit depth, fraction) |
| `pos_params_clean.npy` | (98725, 9) | planet parameters per spectrum |
| `wavelength_binned.npy` | (52,) | wavelength grid (µm, descending) |

## Preprocessing (`data.py`)

1. **Feature-height cut:** `max(spec) - min(spec) >= 210 ppm` keeps 44,026 of 98,725 spectra.
2. **Z-normalisation per spectrum:** `(spec - mean(spec)) / std(spec)`.
3. **Split:** 70 / 15 / 15 train / val / test (seed 42), giving 30,818 / 6,604 / 6,604.

The per-spectrum mean and std and the original row indices are kept, so you can undo the normalisation and match spectra back to `pos_params_clean.npy`.

## Model (`model.py`)

`52-128-64-32-16-8-16-32-64-128-52`, with GELU activations. The bottleneck and the output layer are linear. 35,612 parameters.

## Training (`train.py`)

```
pip install -r requirements.txt
python train.py --out runs/ae_baseline
```

Defaults:
- **Loss:** reconstruction MSE
- **Optimiser:** AdamW, lr 1e-3, weight decay 1e-5, batch size 256
- **Learning rate:** ReduceLROnPlateau (factor 0.5, patience 7)
- **Epochs:** up to 1000, with early stopping on val loss (patience 20) that restores the best weights

Outputs go to the run directory:
- `model.pt`
- `metrics.json`
- `history.json`
- `split_indices.npz`
- `loss.png`
- `reconstructions.png`

## Baseline result (`runs/ae_baseline`)

| | MSE (z-normalised) |
|---|---|
| val (best) | 1.98e-4 |
| test (mean) | 1.96e-4 |
| test (median) | 8.9e-5 |
| test (95th percentile) | 5.9e-4 |

The run used all 1000 epochs, and the best epoch was 996. The val loss kept improving by tiny amounts while the learning rate decayed, so early stopping never triggered. It would trigger with a `min_delta` threshold.

## PCA analysis for the bottleneck size (`pca_analysis.py`)

PCA is fitted on the training split and evaluated on the test split. Results are in `runs/pca/`.

| k | cumulative variance | PCA test MSE |
|---|---|---|
| 4 | 0.861 | 5.4e-2 |
| 6 | 0.932 | 2.6e-2 |
| 8 | 0.966 | 1.3e-2 |
| 10 | 0.980 | 7.6e-3 |
| 12 | 0.988 | 4.4e-3 |
| 13 | 0.992 | 3.0e-3 |
| 16 | 0.996 | 1.6e-3 |

Number of components needed:
- 99% of the variance: 13
- 99.9%: 25
- 99.99%: 39

The scree plot has its clearest break after component 13. The autoencoder with an 8-dimensional bottleneck has a test MSE of 2.0e-4, about 8x lower than PCA with 16 components. So the spectra lie on a nonlinear manifold, and PCA overestimates the number of latent dimensions needed. Treat the PCA value (about 13) as an upper bound for the bottleneck size.

## Bottleneck sweep (`sweep.py`)

The autoencoder was trained with bottleneck sizes 4 to 16, using one seed, the default hyperparameters and early stopping with `--min-delta 1e-3`. Results are in `runs/sweep/`.

| latent | epochs | test MSE | median | p95 |
|---|---|---|---|---|
| 4 | 320 | 3.37e-3 | 8.6e-4 | 1.40e-2 |
| 5 | 358 | 1.31e-3 | 3.1e-4 | 4.8e-3 |
| 6 | 395 | 4.37e-4 | 1.4e-4 | 1.5e-3 |
| 7 | 359 | 5.25e-4 | 1.7e-4 | 1.8e-3 |
| **8** | 747 | **1.96e-4** | 8.9e-5 | 5.9e-4 |
| 9 | 648 | 1.64e-4 | 7.8e-5 | 4.9e-4 |
| 10 | 726 | 1.97e-4 | 9.5e-5 | 6.1e-4 |
| 11 | 841 | 2.12e-4 | 9.8e-5 | 6.5e-4 |
| 12 | 356 | 4.90e-4 | 1.7e-4 | 1.6e-3 |
| 13 | 1000 | 1.83e-4 | 8.2e-5 | 5.7e-4 |
| 14 | 661 | 2.41e-4 | 8.9e-5 | 7.2e-4 |
| 15 | 579 | 1.80e-4 | 8.8e-5 | 5.5e-4 |
| 16 | 720 | 1.78e-4 | 8.5e-5 | 5.3e-4 |

- The error drops by about 20x from 4 to 8 latent dimensions.
- From 8 upwards it levels off at about 1.6 to 2.4e-4, and that variation is within single-seed noise.
- The 7 and 12 runs stopped early (about 350 epochs), apparently on a temporary plateau in the loss.

Chosen bottleneck: **8**, the smallest size on the plateau. This is consistent with the 9 parameters the spectra were generated from.

## Activation sweep (`--param activation`)

Each activation was trained with bottleneck 8 and 3 seeds (42, 43, 44), using default hyperparameters. Results are in `runs/sweep_activation/`.

```
python sweep.py --param activation --values relu leaky_relu elu selu gelu silu mish tanh softplus \
    --seeds 42 43 44 --out runs/sweep_activation
```

| activation | test MSE mean ± std | best seed | median (mean) | epochs (mean) |
|---|---|---|---|---|
| **gelu** | **2.73e-4 ± 1.1e-4** | 1.96e-4 | 1.12e-4 | 571 |
| tanh | 3.38e-4 ± 1.8e-4 | 2.34e-4 | 1.34e-4 | 715 |
| silu | 3.58e-4 ± 1.4e-4 | 2.01e-4 | 1.25e-4 | 601 |
| mish | 3.69e-4 ± 1.6e-4 | 1.83e-4 | 1.26e-4 | 586 |
| elu | 3.80e-4 ± 1.7e-4 | 2.59e-4 | 1.57e-4 | 420 |
| selu | 4.34e-4 ± 3.6e-5 | 3.93e-4 | 2.31e-4 | 406 |
| leaky_relu | 4.60e-4 ± 3.2e-5 | 4.31e-4 | 2.43e-4 | 511 |
| softplus | 5.87e-4 ± 2.4e-5 | 5.67e-4 | 2.01e-4 | 409 |
| relu | 8.44e-4 ± 3.0e-4 | 6.56e-4 | 4.56e-4 | 579 |

- GELU has the best mean and median, so it stays the default.
- ReLU is clearly the worst.
- The smooth activations (GELU, Tanh, SiLU, Mish, ELU) all reach about 2e-4 on their best seed and overlap within seed noise.
- Within each of these smooth activations, the spread between seeds is mostly driven by when early stopping fires. Runs that stopped before about 450 epochs end near 4–6e-4, and runs that went past about 750 epochs reach about 2e-4. A longer `--patience` would probably narrow the gaps between seeds.
