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

### Scoring against known error bars (`chi2_score.py`, `runs/anomaly/chi2/`)

For a noisy spectrum y with per-bin errors σ:
1. z-normalise it.
2. Reconstruct it with the code's autoencoder.
3. Map the reconstruction back to transit depth, `ŷ = AE(z)·s + m`, where m and s are the spectrum's mean and std.
4. Score it with the reduced χ², `Σ((y − ŷ)/σ)² / 52`.

Details:
- The flag threshold is the 99% point of a χ² with 52 degrees of freedom: reduced χ² > 1.51, a nominal 1% false-positive rate.
- The models are the noiseless-trained anomaly models; they are not retrained.
- Noise is white Gaussian, the same σ in every bin, with 2 realisations per level.
- The **oracle** row scores each spectrum against the true noiseless spectrum of the training code for the same planet. It is the best any single-spectrum test could do if the planet's parameters were known.

Mean AUC over the 6 code pairs:

| σ (ppm) | 2 | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| z-norm MSE (old score) | **0.91** | 0.75 | 0.63 | 0.56 | 0.53 | 0.52 | 0.51 |
| AE χ² vs error bars | 0.83 | **0.79** | **0.74** | **0.67** | **0.63** | **0.60** | **0.56** |
| oracle χ² (ceiling) | 1.00 | 0.98 | 0.95 | 0.88 | 0.82 | 0.75 | 0.66 |

At the reduced-χ² > 1.51 threshold:

| σ (ppm) | 2 | 5 | 10 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| own-code spectra flagged (AE; nominal 1%) | 39–47% | 19–25% | 12–16% | 8–10% | 6–8% | 6–7% | 8–9% |
| other-code spectra flagged (AE) | 92–96% | 68–74% | 48–51% | 29–34% | 22–27% | 16–20% | 14–17% |
| other-code spectra flagged (oracle; own-code ~1%) | 98–100% | 81–100% | 61–94% | 42–74% | 32–60% | 23–45% | 14–27% |

- **The χ² score beats z-norm MSE at every noise level from 5 ppm up.** It roughly doubles the margin above chance at 10–50 ppm: AUC 0.74 vs 0.63 at 10 ppm, and 0.60 vs 0.52 at 50 ppm. z-norm MSE is better only at 2 ppm, where the noise is negligible.
- **The score is not yet calibrated.** A typical own-code spectrum has reduced χ² ≈ 1, as it should. But the own-code distribution has a long tail, so 6–47% are flagged instead of 1%.
  - At low σ, the tail is the autoencoder's own reconstruction error, which is large compared with tiny error bars.
  - At high σ, the noisy input is unlike anything the model was trained on, so the reconstruction partly follows the noise.
- **The autoencoder is the bottleneck, not the data.** The oracle reaches AUC 0.88 at 20 ppm and 0.75 at 50 ppm, against 0.67 and 0.60 for the autoencoder. The hardest pair is petitRADTRANS vs TauREx: oracle AUC 0.78 at 20 ppm.
- **Next steps:**
  - Train with noise added to the inputs and clean spectra as targets (a denoising autoencoder), so the reconstruction ignores the noise.
  - Set the threshold empirically from simulated noisy own-code spectra, not from the theoretical χ² distribution.

### Denoising autoencoders (`train.py --noise-ppm`, `runs/anomaly_noise/`)

One model was trained per code and per noise level (10, 20, 30, 50 ppm), giving 12 models.
- **Inputs:** noisy spectra, z-normalised by their own mean and std. A fresh noise draw is used every batch.
- **Targets:** the clean spectra, scaled by the same mean and std.
- **Settings:** otherwise the same as the anomaly models (wide architecture, cosine schedule, patience 50, split seed 42).
- **Stopping:** the noisy validation loss is flat, so 9 of the 12 runs stopped early, after 230–410 epochs.

Launched by `runs/anomaly_noise/run.sh`. Evaluated with:

```
python chi2_score.py --sigmas 10 20 30 50 --out runs/anomaly_noise/chi2
```

The test spectra get noise at the model's training level, and they are scored with reduced χ² against the error bars. Two thresholds are compared:
- **Theoretical:** reduced χ² > 1.51.
- **Empirical:** the 99th percentile of reduced χ² on noisy own-code validation spectra, so the false-positive rate really is about 1%.

Means over the 6 code pairs:

| σ (ppm) | model | AUC | flagged at theoretical threshold: own / other | flagged at empirical 1% threshold: other |
|---|---|---|---|---|
| 10 | noiseless | **0.73** | 14% / 49% | **4.1%** |
| | denoising | 0.68 | 22% / 49% | 2.5% |
| | oracle | 0.95 | 1% / 81% | — |
| 20 | noiseless | **0.67** | 9% / 32% | **4.1%** |
| | denoising | 0.63 | 15% / 32% | 2.5% |
| | oracle | 0.88 | 1% / 62% | — |
| 30 | noiseless | **0.63** | 8% / 24% | **4.1%** |
| | denoising | 0.60 | 12% / 25% | 2.1% |
| | oracle | 0.82 | 1% / 49% | — |
| 50 | noiseless | **0.60** | 7% / 18% | **4.2%** |
| | denoising | 0.56 | 11% / 18% | 1.6% |
| | oracle | 0.75 | 1% / 36% | — |

How close each model's reconstruction is to the true clean spectrum, RMS in ppm, own-code test spectra (median / 99th percentile):

| code | σ | noiseless AE | denoising AE |
|---|---|---|---|
| POS | 10 | 5.5 / 135 | 4.9 / 209 |
| POS | 20 | 10.5 / 136 | 7.7 / 153 |
| POS | 50 | 26.9 / 138 | 18.1 / 375 |
| TauREx | 10 | 5.4 / 85 | 4.4 / 169 |
| TauREx | 20 | 10.5 / 86 | 8.4 / 264 |
| TauREx | 50 | 27.2 / 89 | 18.0 / 375 |

- **Denoising training did not improve anomaly detection.** It is slightly worse than the noiseless models at every noise level.
  - For typical spectra it does denoise better: the median reconstruction error is 20–35% lower.
  - But its worst cases are much worse: the 99th-percentile error rises to 150–375 ppm, against 85–140 ppm. That tail sets the threshold and raises the false-positive rate. Stopping early probably contributes.
- **At a properly calibrated 1% false-positive rate, both autoencoders flag only 2–4% of other-code spectra at 10–50 ppm.** The oracle flags 36–81%. The theoretical-threshold detection rates (18–49%) were inflated by 7–22% false positives.
- **The limit is each autoencoder's worst-case reconstruction error, about 100–375 ppm for its hardest own-code spectra.** That is far larger than the typical difference between codes (about 10 ppm). The hardest spectra are the cool and low-gravity planets. The rate stays near 4% at every noise level because, for the hardest spectra, the model's own error is much larger than the noise.

## Random-forest code classifier (`rf_classifier.py`, `runs/rf_classifier/`)

A 3-class random forest predicts which code produced a spectrum.

**Data:** the same planets and split as the anomaly experiment.
- 210 ppm cut applied in all codes; split seed 42.
- Every planet gives one spectrum per code, so there are 92,157 training spectra and 19,749 test spectra.
- Test planets are never seen in training, in any code.
- **Noise:** white noise of σ ppm is added to the raw spectra, for both training and test, and each spectrum is then z-normalised.

**Settings:**
- 300 trees, `min_samples_leaf=2`, `max_features=sqrt`.
- One noisy copy of each training spectrum.
- Trained and tested at the same σ.

```
python rf_classifier.py                       # σ = 0, 10, 20, 30, 50 ppm
python rf_classifier.py --train-copies 3      # 3 noise draws per training spectrum
```

| σ (ppm) | accuracy (chance 0.33) | POS recall | petitRADTRANS recall | TauREx recall | macro one-vs-rest AUC | mean pairwise AUC |
|---|---|---|---|---|---|---|
| 0 | **0.968** | 0.98 | 0.95 | 0.98 | 0.997 | 0.999 |
| 10 | **0.783** | 0.82 | 0.73 | 0.79 | 0.925 | 0.945 |
| 20 | **0.684** | 0.74 | 0.63 | 0.69 | 0.861 | 0.886 |
| 30 | **0.617** | 0.67 | 0.56 | 0.61 | 0.811 | 0.838 |
| 50 | **0.549** | 0.61 | 0.50 | 0.54 | 0.749 | 0.775 |

Training on 3 noise draws per spectrum adds only about 1 point of accuracy: 0.692 at 20 ppm and 0.561 at 50 ppm.

The pairwise AUC is computed on test spectra of codes a and b with the score p_b / (p_a + p_b), so it can be compared directly with the anomaly-detection AUCs (mean over the 6 pairs):

| σ (ppm) | 10 | 20 | 30 | 50 |
|---|---|---|---|---|
| AE χ² (noiseless models) | 0.73 | 0.67 | 0.63 | 0.60 |
| "oracle" χ² against the own code's true spectrum | 0.95 | 0.88 | 0.82 | 0.75 |
| **random forest** | **0.945** | **0.886** | **0.838** | **0.775** |

- **The classifier separates the codes far better than the autoencoders at every noise level.** It matches or slightly beats the "oracle". The oracle was only a one-class goodness-of-fit test: is this spectrum consistent with code a? The classifier instead learns the specific ways the codes differ, so it is not the ceiling for a test that knows all three codes.
- **The noiseless forest overfits slightly** (train accuracy 1.00), but test accuracy is still 0.97.
- **petitRADTRANS is the hardest code to identify** at every noise level, and POS the easiest.
- **The most informative bins** are about 1.27 µm, 1.55 µm, 2.9 µm, 2.35 µm and 6.3 µm (see `rf_importance.png`).

### Accuracy by planet type (`rf_by_planet.py`, `runs/rf_classifier/by_planet/`)

This uses the test-set probabilities saved by `rf_classifier.py`. A planet's accuracy is the fraction of its three spectra (one per code) that are attributed to the right code. The table compares the lowest and highest of 8 quantile bins:

| quantity | σ = 0 | σ = 10 | σ = 20 | σ = 30 | σ = 50 ppm |
|---|---|---|---|---|---|
| code difference RMS (ppm) | 0.95 → 0.94 | **0.50 → 0.92** | **0.40 → 0.92** | **0.38 → 0.89** | **0.35 → 0.85** |
| feature height (ppm) | 0.96 → 0.93 | **0.58 → 0.89** | **0.44 → 0.89** | **0.41 → 0.85** | **0.37 → 0.81** |
| log g | 0.91 → 0.98 | 0.85 → 0.69 | 0.81 → 0.58 | 0.77 → 0.52 | 0.72 → 0.45 |
| Mp | 0.91 → 0.98 | 0.82 → 0.73 | 0.76 → 0.61 | 0.72 → 0.55 | 0.67 → 0.48 |
| T | 0.95 → 0.98 | 0.76 → 0.86 | 0.63 → 0.78 | 0.57 → 0.71 | 0.50 → 0.63 |
| chem1 | 0.94 → 0.99 | 0.73 → 0.87 | 0.65 → 0.77 | 0.57 → 0.69 | 0.51 → 0.61 |
| chem2 | 0.98 → 0.96 | 0.81 → 0.65 | 0.73 → 0.57 | 0.66 → 0.50 | 0.59 → 0.45 |
| chem5 | 0.97 → 0.97 | 0.80 → 0.74 | 0.71 → 0.63 | 0.65 → 0.53 | 0.57 → 0.49 |
| Rs | 0.96 → 0.97 | 0.84 → 0.74 | 0.75 → 0.63 | 0.70 → 0.57 | 0.60 → 0.51 |
| Rp | 0.96 → 0.96 | 0.75 → 0.82 | 0.62 → 0.73 | 0.57 → 0.67 | 0.51 → 0.59 |
| chem3, chem4 | flat | ≤ 0.05 change | ≤ 0.05 change | ≤ 0.06 change | ≤ 0.06 change |

Accuracy for planets whose codes differ by at least X ppm RMS:

| X | share of planets | σ = 0 | 10 | 20 | 30 | 50 ppm |
|---|---|---|---|---|---|---|
| 10 | 51% | 0.97 | 0.90 | 0.83 | 0.77 | 0.69 |
| 30 | 24% | 0.95 | 0.92 | 0.90 | 0.87 | 0.81 |
| 50 | 16% | 0.95 | 0.92 | 0.91 | 0.89 | 0.85 |
| 100 | 9% | 0.93 | 0.91 | 0.91 | 0.88 | 0.85 |

- **Noiseless:** accuracy is 0.91–0.99 for every planet type. Only the lowest-mass, lowest-gravity planets dip slightly, to 0.91.
- **With noise, signal size dominates.**
  - Planets whose codes differ by about 30 ppm or more (about a quarter of the sample) are still classified 81–90% correctly at 30–50 ppm noise.
  - Planets whose codes differ by under about 5 ppm fall to near chance.
  - Feature height tracks the same thing, because larger features give larger absolute differences between codes.
- **The planet properties that help under noise are the ones that give large features:**
  - low gravity and low Mp: 0.72 vs 0.45 at 50 ppm;
  - hot planets: T above about 2200 K;
  - large Rp;
  - small host stars (Rs);
  - high chem1.
- **Planets that are harder to classify under noise:**
  - planets around 1500 K;
  - high chem2 or chem5 (the abundances anti-correlated with T).
- **This is the opposite of the autoencoder.** Low-gravity planets were the autoencoder's hardest to reconstruct, but they are the easiest to classify once there is noise. Their spectra are large and complex. Big features make the code differences stand out above the noise, but complex shapes are harder for an 8-dimensional bottleneck to reproduce.

## Paired residual analysis (`residual_analysis.py`, `runs/residuals/`)

Every planet has a noiseless spectrum from each code, so the code systematics are measured directly as difference spectra for the same planet. The analysis uses the 43,885 planets that pass the 210 ppm cut in all codes. Each difference is split into:
- **Offset:** the wavelength-mean difference. It is degenerate with Rp in a fit, so it is reported separately and removed.
- **Shape residual:** the rest, in ppm ("absolute") and as a fraction of the pair's mean feature height ("relative").

Each pair's residuals then go through PCA. Each component score is correlated (Spearman) with the parameters and regressed on the 9 parameters with a random forest (out-of-sample R²).

```
python residual_analysis.py              # --no-cut for all planets, --n-components N
```

| pair | offset (median, 16–84%) | shape residual RMS (median / 90th pct) | relative to feature height | relative PCA: comps for 90% / 99% | R² of relative PC1–4 from parameters |
|---|---|---|---|---|---|
| POS − petitRADTRANS | −24 ppm (−109 to −9) | 10 / 91 ppm | 1.3% | 6 / 22 | 0.99, 0.96, 0.96, 0.95 |
| POS − TauREx | −21 ppm (−86 to −8) | 10 / 75 ppm | 1.3% | 7 / 24 | 0.98, 0.94, 0.96, 0.95 |
| petitRADTRANS − TauREx | +2 ppm (−1 to +21) | 8 / 75 ppm | 1.0% | 6 / 21 | 0.99, 0.96, 0.94, 0.96 |

- **POS gives systematically shallower spectra than the other two codes.** The offset is about −20 to −25 ppm on median, and its size scales almost exactly with the feature height (Spearman −0.96 for POS − petitRADTRANS, −0.89 for POS − TauREx). That points to a different reference radius, reference pressure or scale-height treatment in POS. petitRADTRANS and TauREx agree on the offset to about 2 ppm.
- **After the offset is removed, the codes agree to about 1–1.3% of the feature height** (median 8–10 ppm). The worst 10% of planets differ by 75–90 ppm; these are the large-feature planets.
- **petitRADTRANS has two narrow, localised differences, at the 1.55 µm and 2.36 µm bins** (plus 2.31 µm). They dominate PC1 of both pairs that involve petitRADTRANS (55–60% of the variance). The 2.3–2.4 µm region holds the CO and CH4 bands, and 1.5 µm is near an NH3 / H2O band. This suggests an opacity or line-list difference there. Their strength tracks T and chem4 and anti-tracks chem1.
- **POS − TauREx is dominated by a broad short-wavelength difference** (0.55–1.3 µm, PC1 60%), where POS is lower. It is driven almost entirely by chem1 (ρ = −0.78), consistent with a different treatment of the continuum (Rayleigh / collision-induced absorption) or of chem1's short-wavelength opacity. POS also shows a slight positive tilt towards long wavelengths (6–7.3 µm) against both other codes.
- **The relative systematics are a near-deterministic function of the parameters.** Each of the first 4 relative components is predicted from the 9 parameters with R² ≥ 0.94. So the code differences could be emulated and corrected, or added as a modelled error term in retrievals. The absolute (ppm) components are less predictable (R² 0.1–0.65), because their size also scales with the feature amplitude.
- **These are the same bins the random forest relies on most:** 1.27, 1.55 and 2.35 µm.

Plots:
- `residual_spectra.png`: residual bands vs wavelength.
- `pca_components_{absolute,relative}.png`: component shapes, with approximate band positions.
- `pca_drivers_{absolute,relative}.png`: parameter correlations.
- `offsets.png`: offset over T and log g.

## ML retrieval with XGBoost: self vs cross retrieval (`ml_retrieval.py`, `runs/retrieval/`)

For each training code, one XGBoost regressor per parameter is trained on that code's spectra (noiseless). It is then applied to the same held-out test planets in every code:
- **Self retrieval:** the training code's own spectra.
- **Cross retrieval:** the other codes' spectra.

The planets are identical, so the change from self to cross predictions is caused only by the code differences.

**Data:** the 210 ppm cut in all codes and the shared split (30,719 / 6,583 / 6,583).

**Features:** the z-normalised spectrum (52 bins) plus log10 mean depth and log10 std (these keep Rp/Rs).

**XGBoost settings:** learning rate 0.1, max depth 8, subsample 0.8, colsample 0.8, up to 3000 rounds, early stopping after 100 rounds on val RMSE.

**Learning curves:** retrained on 5, 10, 25 and 50% of the training planets.

```
python ml_retrieval.py                       # trains 27 models + learning curves (~40 min)
python ml_retrieval.py --noise-ppm 20 --out runs/retrieval_20ppm
python ml_retrieval.py --replot              # redo plots from saved predictions
```

Test R² (diagonal = self retrieval):

| trained on → tested on | Rs | Rp | Mp | T | chem1 | chem2 | chem3 | chem4 | chem5 |
|---|---|---|---|---|---|---|---|---|---|
| POS → POS | 0.77 | 0.62 | 0.90 | 0.997 | 0.96 | 0.97 | 0.85 | 0.994 | 0.90 |
| POS → petitRADTRANS | 0.63 | 0.37 | 0.86 | 0.970 | 0.95 | 0.95 | **0.54** | 0.987 | 0.89 |
| POS → TauREx | 0.55 | **0.24** | 0.84 | 0.989 | 0.96 | 0.95 | **0.59** | 0.978 | 0.86 |
| petitRADTRANS → POS | 0.70 | 0.51 | 0.87 | **0.924** | 0.95 | 0.94 | 0.68 | 0.984 | 0.89 |
| petitRADTRANS → petitRADTRANS | 0.76 | 0.61 | 0.90 | 0.997 | 0.96 | 0.97 | 0.85 | 0.994 | 0.90 |
| petitRADTRANS → TauREx | 0.68 | 0.50 | 0.88 | 0.950 | 0.96 | 0.96 | 0.71 | 0.983 | 0.83 |
| TauREx → POS | 0.68 | 0.47 | 0.86 | 0.970 | 0.95 | 0.95 | **0.62** | 0.977 | 0.89 |
| TauREx → petitRADTRANS | 0.72 | 0.54 | 0.88 | 0.977 | 0.95 | 0.96 | **0.62** | 0.989 | 0.88 |
| TauREx → TauREx | 0.76 | 0.60 | 0.90 | 0.997 | 0.96 | 0.96 | 0.86 | 0.994 | 0.90 |

Median bias (predicted − true) caused by switching code (self-retrieval biases are all within ±1% of the range):

| trained on → tested on | T (K) | chem3 (dex) | Rp (R_J) | Rs (R☉) |
|---|---|---|---|---|
| POS → petitRADTRANS / TauREx | +39 / +39 | +0.29 / +0.22 | −0.078 / −0.044 | −0.052 / −0.025 |
| petitRADTRANS → POS / TauREx | −45 / −2 | −0.29 / +0.03 | +0.056 / +0.021 | +0.038 / +0.017 |
| TauREx → POS / petitRADTRANS | −28 / +9 | −0.37 / −0.05 | +0.039 / −0.018 | +0.027 / −0.021 |

- **Self retrieval is accurate for T and most abundances:**
  - T: R² 0.997.
  - chem1, chem2 and chem4: R² 0.96–0.99.
  - Mp and chem5: about 0.90.
  - chem3: 0.85.
  - Rs and Rp are only partly constrained (R² 0.76 and 0.61), because the transit depth mainly fixes their ratio (Rp/Rs)².
- **Cross retrieval adds code-dependent biases.**
  - **T:** POS spectra retrieved with a petitRADTRANS- or TauREx-trained model come out 28–45 K cooler (median). The bias grows with T, reaching about −300 K at 3000 K for petitRADTRANS → POS (`bias_vs_true.png`). Going the other way, the POS-trained model makes the other codes about 39 K hotter. petitRADTRANS and TauREx are mutually consistent to within 10 K.
  - **chem3 is the most code-sensitive parameter.** Its R² drops from 0.85 to 0.54–0.62 whenever POS is involved, with ±0.2–0.4 dex median bias and errors of 1–2 dex at low abundance.
  - **Rp and Rs** shift by about 2–4% of their range, consistent with POS's shallower overall depth (see the residual analysis). Rp R² drops from 0.62 to 0.24–0.37 for POS-trained models.
  - **chem1, chem2, chem4, chem5 and Mp are robust:** R² changes by ≤ 0.05, and biases are ≤ 1% of the range.
- **The code systematics are a floor that more training data does not remove.** In the learning curves, self-retrieval R² keeps rising with the number of training planets. Cross-retrieval R² flattens or even falls (Rp and Rs for POS-trained models, T for petitRADTRANS-trained models), because the model becomes more specialised to its own code.
- **Most models are overfitting and not fully converged.** Train RMSE keeps falling while val RMSE flattens early (`loss_curves_*.png`). Many models used all 3000 rounds for very small val gains, so stronger regularisation would cut training time with little loss in accuracy.
- **Plots:**
  - `true_vs_pred_<code>.png`
  - `residual_spread.png`
  - `code_bias.png` (bias matrix, % of range)
  - `bias_vs_true.png`
  - `learning_curves.png`
  - `loss_curves_<code>.png`

### Noiseless-trained retrieval tested on 10 ppm spectra (`runs/retrieval_test10ppm/`)

The same 27 noiseless-trained XGBoost models are applied to the test spectra with 10 ppm white noise added per bin. The models were retrained with the same settings and seed, and they reproduce the noiseless results exactly. They are saved in `runs/retrieval_models/`, which is git-ignored.

```
python ml_retrieval.py --test-noise-ppm 10 --models-dir runs/retrieval_models --train-fracs \
    --out runs/retrieval_test10ppm
python compare_retrieval.py runs/retrieval runs/retrieval_test10ppm --labels "noiseless test" "10 ppm test"
```

R² for noiseless → 10 ppm test spectra:

| parameter | self retrieval | cross retrieval (range over the 6 cross pairs) |
|---|---|---|
| T | 0.997 → 0.96–0.97 | 0.92–0.99 → 0.89–0.96 |
| chem4 | 0.994 → 0.96–0.97 | 0.98–0.99 → 0.95–0.96 |
| chem1 | 0.96 → 0.89 | 0.95–0.96 → 0.88–0.89 |
| chem2 | 0.96–0.97 → 0.90–0.91 | 0.94–0.96 → 0.87–0.90 |
| Mp | 0.90 → 0.85–0.86 | 0.84–0.88 → 0.81–0.85 |
| chem5 | 0.90 → 0.79 | 0.83–0.89 → 0.72–0.80 |
| chem3 | 0.85–0.86 → 0.57–0.64 | 0.54–0.71 → 0.39–0.52 |
| Rs | 0.76–0.77 → 0.56–0.59 | 0.55–0.72 → 0.44–0.57 |
| Rp | 0.60–0.62 → 0.33–0.37 | 0.24–0.54 → 0.09–0.34 |

Code biases (median predicted − true) are essentially unchanged by the noise:

| trained → tested | T (K) | chem3 (dex) | Rp (R_J) |
|---|---|---|---|
| POS → petitRADTRANS | +39 → +38 | +0.29 → +0.26 | −0.078 → −0.070 |
| POS → TauREx | +39 → +31 | +0.22 → +0.21 | −0.044 → −0.044 |
| petitRADTRANS → POS | −45 → −63 | −0.29 → −0.29 | +0.056 → +0.056 |
| TauREx → POS | −28 → −34 | −0.37 → −0.39 | +0.039 → +0.042 |

The 16–84% scatter of predicted − true roughly doubles for T, chem1, chem2 and chem4. For example, T self retrieval goes from about 27 K to 83–89 K.

- **Noise adds scatter, not bias.** All R² values drop. The code-induced median biases stay almost exactly where they were, e.g. T +39 K, chem3 ±0.3 dex and Rp −0.07 R_J. So the code systematics are an additive bias that 10 ppm noise does not wash out.
- **The bias relative to the scatter shrinks but does not vanish.**
  - T bias for POS → petitRADTRANS is about 0.3× the 16–84% half-width (38 vs 128 K).
  - chem3 bias is about 0.25× (0.26 vs 1.1 dex).
  - For a population of planets, these biases would still be clearly visible.
- **The self-retrieval advantage remains for chem3, Rp and T involving POS.** chem3 self R² is 0.57–0.64 against 0.39–0.52 for cross. For the robust parameters (chem1, chem2, chem4, Mp), self and cross are within noise of each other.
- **These models never saw noise in training, so the noisy-test R² values are a lower bound.** Training with noise (`--noise-ppm 10`) should recover part of the lost precision.
- **Plots:**
  - `comparison.png`: R², bias and spread, noiseless vs 10 ppm, for all train→test combinations.
  - `true_vs_pred_<code>.png`
  - `residual_spread.png`
  - `code_bias.png`
  - `bias_vs_true.png`

### Noiseless-trained retrieval vs test noise: 0, 10, 20, 30, 50 ppm (`runs/retrieval_noise/`)

The same saved models are tested at each noise level (`runs/retrieval_test{10,20,30,50}ppm/`), and the results are summarised with:

```
for s in 20 30 50; do python ml_retrieval.py --test-noise-ppm $s --models-dir runs/retrieval_models \
    --train-fracs --out runs/retrieval_test${s}ppm; done
python compare_retrieval.py runs/retrieval runs/retrieval_test{10,20,30,50}ppm --out runs/retrieval_noise
```

Mean R², self / cross (cross = mean over the 6 cross pairs):

| parameter | 0 ppm | 10 ppm | 20 ppm | 30 ppm | 50 ppm |
|---|---|---|---|---|---|
| T | 0.997 / 0.963 | 0.962 / 0.931 | 0.894 / 0.867 | 0.819 / 0.794 | 0.678 / 0.659 |
| chem4 | 0.994 / 0.983 | 0.964 / 0.954 | 0.916 / 0.907 | 0.866 / 0.859 | 0.769 / 0.764 |
| chem2 | 0.966 / 0.951 | 0.900 / 0.891 | 0.831 / 0.823 | 0.772 / 0.765 | 0.680 / 0.674 |
| chem1 | 0.959 / 0.953 | 0.888 / 0.884 | 0.797 / 0.792 | 0.707 / 0.705 | 0.541 / 0.538 |
| Mp | 0.901 / 0.866 | 0.856 / 0.834 | 0.792 / 0.776 | 0.727 / 0.716 | 0.608 / 0.600 |
| chem5 | 0.901 / 0.873 | 0.790 / 0.774 | 0.641 / 0.631 | 0.507 / 0.502 | 0.296 / 0.291 |
| chem3 | 0.851 / 0.627 | 0.596 / 0.438 | 0.287 / 0.163 | 0.001 / −0.094 | −0.459 / −0.533 |
| Rs | 0.761 / 0.661 | 0.578 / 0.525 | 0.370 / 0.342 | 0.204 / 0.190 | −0.010 / −0.015 |
| Rp | 0.609 / 0.437 | 0.349 / 0.250 | 0.035 / −0.028 | −0.222 / −0.265 | −0.572 / −0.598 |

Median bias ± 16–84% half-width:

| trained → tested, parameter | 0 ppm | 10 ppm | 20 ppm | 30 ppm | 50 ppm |
|---|---|---|---|---|---|
| self, T (K; mean of 3 codes) | 0 ± 27 | −5 ± 86 | −13 ± 149 | −21 ± 208 | −41 ± 305 |
| POS → petitRADTRANS, T (K) | +39 ± 86 | +38 ± 128 | +31 ± 178 | +24 ± 235 | +9 ± 323 |
| petitRADTRANS → POS, T (K) | −45 ± 143 | −63 ± 179 | −82 ± 224 | −100 ± 265 | −130 ± 333 |
| TauREx → POS, T (K) | −28 ± 58 | −34 ± 116 | −46 ± 179 | −55 ± 232 | −80 ± 316 |
| self, chem3 (dex; mean) | −0.03 | −0.04 | −0.07 | −0.12 | −0.22 |
| POS → petitRADTRANS, chem3 (dex) | +0.29 ± 0.93 | +0.26 ± 1.10 | +0.23 ± 1.32 | +0.19 ± 1.53 | +0.09 ± 1.86 |
| TauREx → POS, chem3 (dex) | −0.37 ± 0.78 | −0.39 ± 1.04 | −0.43 ± 1.30 | −0.49 ± 1.52 | −0.60 ± 1.83 |
| POS → petitRADTRANS, Rp (R_J) | −0.078 ± 0.25 | −0.070 ± 0.30 | −0.056 ± 0.35 | −0.058 ± 0.38 | −0.046 ± 0.43 |
| petitRADTRANS → POS, Rp (R_J) | +0.056 ± 0.24 | +0.056 ± 0.29 | +0.057 ± 0.34 | +0.058 ± 0.38 | +0.065 ± 0.43 |

- **Precision falls steeply with noise, and noise-free training makes it worse.**
  - At 30–50 ppm, R² is negative for Rp, chem3 and Rs, so the models do worse than predicting the mean value. The models have never seen noise in training, so their predictions on noisy spectra are unreliable.
  - T (0.68), chem4 (0.77), chem2 (0.68) and Mp (0.61) keep some skill at 50 ppm.
- **Noise also adds its own bias.** Self-retrieval T drifts to about −40 K at 50 ppm, and chem3 to about −0.2 dex. In cross retrieval, this noise-induced shift adds to the code bias:
  - Where the two point the same way, the bias grows: petitRADTRANS → POS T goes from −45 to −130 K, and TauREx → POS chem3 from −0.37 to −0.60 dex.
  - Where they point opposite ways, it partly cancels: POS → petitRADTRANS T goes from +39 to +9 K.
- **The code biases themselves stay roughly constant in absolute size, while the scatter grows.** The median ratio |bias| / half-width for cross retrieval falls from about 0.15–0.35 noiseless to about 0.07–0.17 at 50 ppm (`vs_noise.png`, bottom row). So for single planets, the code systematics become a minor part of the error budget beyond about 20–30 ppm. They remain a coherent shift for population studies.
- **The self vs cross R² gap closes with noise for most parameters** (T, Mp and the robust abundances are within 0.02–0.03 by 20 ppm). It persists longest for chem3 (0.85/0.63 at 0 ppm, 0.29/0.16 at 20 ppm) and for Rp.
- **Plots:**
  - `vs_noise.png`: R², |bias|, half-width and |bias|/half-width vs noise, self vs cross.
  - `comparison.png`: every train→test pair at all 5 noise levels.
  - `comparison_table.txt`: the full numbers.
