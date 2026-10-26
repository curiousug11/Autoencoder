# Spectrum autoencoder

Fully connected autoencoder for binned transmission spectra (POS dataset).

## Data (`data/`)

| File | Shape | |
|---|---|---|
| `pos_spec_binned.npy` | (98725, 52) | binned spectra (transit depth, fraction) |
| `pos_params_clean.npy` | (98725, 9) | planet parameters per spectrum |
| `prt_spec_binned.npy` | (98725, 52) | binned spectra from petitRADTRANS |
| `prt_params_clean.npy` | (98725, 9) | parameters for the petitRADTRANS spectra |
| `taurex_spec_binned.npy` | (98725, 52) | binned spectra from TauREx |
| `taurex_params_clean.npy` | (98725, 9) | parameters for the TauREx spectra |
| `wavelength_binned.npy` | (52,) | wavelength grid (µm, descending) |

All three parameter files are identical: row *i* is the same planet in POS, petitRADTRANS and TauREx.

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

## Wider model + cosine schedule (`runs/ae_wide_cosine/`)

Settings:
- **Architecture:** `52-256-128-64-32-16-8`, mirrored for the decoder (115k parameters)
- **Activation:** GELU
- **Learning rate:** cosine schedule from 1e-3 to 1e-6 over 1000 epochs
- **Early stopping:** patience 50
- **Seeds:** 3

```
python sweep.py --param scheduler --values cosine --seeds 42 43 44 --workers 3 --out runs/ae_wide_cosine \
    -- --hidden 256 128 64 32 16 --latent 8 --patience 50
```

| seed | best epoch | val MSE | test MSE | median | p95 |
|---|---|---|---|---|---|
| 42 | 996 | 5.7e-5 | 6.0e-5 | 1.8e-5 | 1.5e-4 |
| 43 | 994 | 5.9e-5 | 7.1e-5 | 2.2e-5 | 2.2e-4 |
| 44 | 996 | 1.2e-4 | 1.1e-4 | 3.0e-5 | 3.7e-4 |
| **mean** | | | **8.0e-5 ± 2.6e-5** | 2.3e-5 | 2.5e-4 |

- The mean test MSE is about 3.4x lower than the previous GELU configuration (`128-64-32-16`, plateau schedule, patience 20), which had 2.73e-4 ± 1.1e-4.
- The median is about 5x lower.
- All seeds ran the full 1000 epochs.
- A gap between train and val loss now appears (about 4e-5 vs 6e-5).

## Comparison: width vs schedule (`runs/ablation/`)

This 2×2 comparison uses seeds 42 and 43 in every configuration, so results can be compared seed for seed. All runs use GELU and a bottleneck of 8. The new runs are launched by `runs/ablation/run.sh`; the other two rows come from earlier runs.

| hidden layers | schedule, patience | seed 42 | seed 43 | mean | source |
|---|---|---|---|---|---|
| 128-64-32-16 | plateau, 20 | 1.96e-4 | 2.20e-4 | 2.08e-4 | `runs/sweep_activation/activation_gelu` |
| 128-64-32-16 | cosine, 50 | 1.13e-4 | 1.07e-4 | 1.10e-4 | `runs/ablation/small_cosine_p50` |
| 256-128-64-32-16 | plateau, 20 | 3.23e-4 * | 1.20e-4 | — | `runs/ablation/wide_plateau_p20` |
| 256-128-64-32-16 | cosine, 50 | 5.97e-5 | 7.13e-5 | 6.55e-5 | `runs/ae_wide_cosine` |

\* Stopped early at epoch 334: the plateau schedule had already lowered the learning rate to 4e-6, and the run stalled.

- **Cosine schedule with patience 50, alone:** about 1.9x lower test MSE (2.08e-4 to 1.10e-4). Both cosine runs train the full 1000 epochs and are consistent across seeds.
- **Wider model, alone:** about 1.8x lower (seed 43: 2.20e-4 to 1.20e-4), as long as training doesn't stall.
- **Both together:** about 3.2x lower. The two effects roughly multiply.
- **The plateau schedule with patience 20 is unreliable.** The learning rate can shrink too quickly, and the run then stops early on a temporary plateau. Cosine with patience 50 avoids this.

## Cross-code anomaly detection (`anomaly.py`, `runs/anomaly/`)

One autoencoder is trained per radiative-transfer code. It is then tested on held-out planets from its own code (normal) and from the other two codes (anomalous).

**Data**
- **Shared feature cut:** a planet is kept only if it passes the 210 ppm cut in all three codes. 43,885 planets remain.
- **Normalisation:** each spectrum is z-normalised on its own.
- **Shared split:** all codes use one train/val/test split of 30,719 / 6,583 / 6,583 planets. Test planets are unseen in every code.

**Model:** `52-256-128-64-32-16-8`, GELU, cosine schedule, patience 50.
- POS and petitRADTRANS were trained with 3 init seeds each (42, 43, 44). TauREx was trained with seed 42 only.
- All runs use the same split (split seed 42).
- For each code, evaluation uses the run with the lowest validation loss on that code's own spectra. No anomalous data is used to choose.

```
python anomaly.py                                              # trains the 3 models in parallel, then evaluates
python anomaly.py --train-datasets pos prt --init-seeds 43 44  # extra seeds, then re-evaluate
python anomaly.py --eval-only
```

Own-code validation MSE for each run (the selected run in bold):

| code | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| POS | 2.28e-4 | 2.30e-4 | **2.00e-4** |
| petitRADTRANS | 2.56e-4 (stopped at epoch 487) | 2.74e-4 (stopped at 802) | **1.51e-4** |
| TauREx | **5.09e-5** | — | — |

**Scoring:** the anomaly score is the per-spectrum reconstruction MSE. Thresholds are the 95th and 99th percentiles of the selected model's error on its own validation spectra.

| trained on | own test MSE (median / mean) | tested on | AUC | flagged at p95 | flagged at p99 |
|---|---|---|---|---|---|
| POS | 3.9e-5 / 2.0e-4 | petitRADTRANS | 0.962 | 83% | 29% |
| | | TauREx | 0.964 | 78% | 34% |
| petitRADTRANS | 2.9e-5 / 1.6e-4 | POS | 0.970 | 88% | 38% |
| | | TauREx | 0.967 | 85% | 33% |
| TauREx | 2.1e-5 / 5.5e-5 | POS | **0.992** | 98% | 79% |
| | | petitRADTRANS | **0.988** | 96% | 78% |

For each model, the fraction of its own test spectra flagged is about 5% at p95 and 1% at p99, as expected.

- **The autoencoders separate the codes clearly in every pairing.** Spectra from other codes have reconstruction errors about 40–80x higher (in median) than spectra from the training code. AUC is between 0.962 and 0.992.
- **Retraining with more seeds improved the POS and petitRADTRANS models.** Their AUCs rose from 0.945–0.956 (seed 42) to 0.962–0.970, and their 95th-percentile flag rates rose by 5–13 points.
- **The TauREx model still separates best.**
  - All three POS seeds plateau at a val MSE of 2.0–2.3e-4, about 4x worse than TauREx.
  - The errors on POS's own spectra have a long upper tail (a shoulder near 1e-3.5) that overlaps the anomalous spectra.
  - petitRADTRANS shows a similar but weaker tail.
  - These spectra appear harder to reconstruct at this model size, rather than the earlier result being bad luck with seeds.

### Which planets are poorly reconstructed? (`poor_recon.py`, `runs/anomaly/poor_recon/`)

For each code, the selected anomaly model scores that code's own val + test spectra (13,166 planets, none used in training). The top 5% by reconstruction MSE (658 planets) count as "poorly reconstructed".

The parameters are Rs, Rp, Mp, T and 5 log abundances (chem1–chem5, in file column order). Three derived quantities are also checked:
- log g = log10(Mp/Rp²), in Jupiter units.
- log H/Rp = log10(T·Rp/Mp), which is proportional to scale height over planet radius at fixed mean molecular weight.
- feature height.

The AUC columns give the probability that a poorly reconstructed planet has a higher value than a random other planet. 0.5 means no difference, and values far from 0.5 mean a strong link.

| quantity | POS AUC | petitRADTRANS AUC | TauREx AUC | median poor vs rest (POS) |
|---|---|---|---|---|
| T | **0.16** | **0.17** | 0.33 | 768 vs 1940 K |
| Mp | **0.24** | **0.23** | **0.11** | 0.39 vs 1.94 |
| log g | **0.24** | **0.22** | **0.17** | −0.58 vs −0.03 |
| log H/Rp | 0.59 | 0.62 | **0.80** | 3.41 vs 3.12 |
| chem5 | 0.62 | 0.63 | 0.55 | −6.9 vs −7.7 |
| chem3 | 0.60 | 0.60 | 0.52 | −5.1 vs −6.1 |
| chem2 | 0.58 | 0.55 | 0.50 | −6.7 vs −7.4 |
| Rp | 0.49 | 0.50 | 0.38 | 1.50 vs 1.50 |
| Rs, chem1, chem4 | 0.41–0.47 | 0.43–0.49 | 0.44–0.52 | |
| feature height | 0.52 | 0.55 | 0.66 | 800 vs 793 ppm |

`poor_recon_T_logg.png` shows the fraction of poorly reconstructed planets in bins of T and log g. Two regions stand out:
- **Cool planets (T below about 600 K), at any gravity.** In POS and petitRADTRANS, more than 40% of these planets are in the worst 5%. They also tend to have higher chem2 and chem5, the two abundances that are anti-correlated with T in the prior.
- **Very low-gravity, puffy planets (log g below about −0.9, so large H/Rp), at any temperature.** This region is the dominant one for TauREx, where low Mp and low Rp both matter.

Other findings:
- **Feature height is not the cause.** Poorly reconstructed spectra are not those close to the 210 ppm cut.
- **The same planets are hard in every code.**
  - POS and petitRADTRANS share 479 of their 658 poor planets; 33 would be expected by chance. The Spearman correlation of their per-planet errors is 0.92.
  - POS and TauREx share 331, and petitRADTRANS and TauREx share 349.
  - 291 planets are in the poor set of all three codes.
  - So the difficulty comes from the planet's parameters, not from the code.
- **Both regions are sparsely sampled in the training set.** Cool planets are partly removed by the feature-height cut, and very low gravity is the edge of the Mp range. This region is what causes the overlap between normal and anomalous errors in the anomaly histograms.

```
python poor_recon.py --top-frac 0.05
```

### Robustness to observational noise (`noise_robustness.py`, `runs/anomaly/noise/`)

The trained anomaly models (no retraining) score the same test planets after white Gaussian noise of σ ppm is added to every bin of the raw spectra. The noisy spectra are then z-normalised as usual. Each noise level uses 2 noise realisations.

| σ (ppm) | 0 | 2 | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| AUC, range over the 6 code pairs | 0.96–0.99 | 0.90–0.92 | 0.73–0.76 | 0.61–0.64 | 0.55–0.57 | 0.52–0.54 | 0.51–0.53 | 0.50–0.52 |

How different the codes are for the same planet: the RMS difference between their mean-subtracted noiseless spectra.

| code pair | median | 10th–90th percentile |
|---|---|---|
| POS vs petitRADTRANS | 11 ppm | 3–93 ppm |
| POS vs TauREx | 10 ppm | 3–76 ppm |
| petitRADTRANS vs TauREx | 9 ppm | 2–77 ppm |

TauREx model vs POS spectra, restricted to planets whose code difference is large:

| planets with RMS difference ≥ | share of planets | AUC at 10 ppm | AUC at 20 ppm | AUC at 50 ppm |
|---|---|---|---|---|
| 0 ppm (all) | 100% | 0.63 | 0.56 | 0.52 |
| 30 ppm | 22% | 0.95 | 0.84 | 0.66 |
| 50 ppm | 14% | 0.98 | 0.93 | 0.75 |
| 100 ppm | 8% | 0.98 | 0.97 | 0.89 |

- **For a typical planet the codes differ by only about 10 ppm RMS.** Noise of that size or larger hides the difference.
- **The noiseless models are not robust to noise.** AUC falls to about 0.75 at 5 ppm and to chance at 30–50 ppm.
- **Detection is still possible for planets where the codes differ strongly.** For the roughly 8–14% of planets with code differences above 50–100 ppm, AUC stays at 0.75–0.97 for noise up to 20–50 ppm.
