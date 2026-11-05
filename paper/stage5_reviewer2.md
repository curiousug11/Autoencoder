# Reviewer 2 critique

**Recommendation:** major revision.

The paired three-code design is a useful idea, and the dataset could become a community reference. In its current form, however, the manuscript's central claims rest on an idealised noise model, under-replicated experiments, an unconverged proxy for retrieval, and physical interpretations the design cannot support. Below are four weaknesses, each with a fix that uses the authors' own codebase.

---

## W1. The detectability conclusions rest on an unrealistic noise model and handicapped detectors

**Problem.**
- Every noise experiment uses white Gaussian noise with the same σ in every bin. Ariel's errors vary strongly with wavelength: the photometric bins, the R ≈ 11/51/16 segments and the long-wavelength channel all differ. Because the inter-code residuals are concentrated in a few bins (1.55, 2.31, 2.36 µm and 0.55–1.27 µm), whether those specific bins are noisy decides detectability. A uniform σ says nothing about this.
- The autoencoders and the XGBoost retrievals are trained without noise and then tested with it, so the noisy results are lower bounds by construction.
- The one "noise-aware" model, the denoising autoencoder, is reported as a negative result even though 10 of its 12 runs early-stopped (between 226 and 885 epochs). The headline claims that "the autoencoder is the bottleneck" and that "the denoising objective is the wrong trade" are not established.

**Fix using the codebase.**
1. Supply an Ariel per-bin error profile, then rerun χ² scoring with it. The scoring script already accepts a per-bin error-shape file through `--sigma-file` and scales it by each `--sigmas` value.
2. Retrain the denoising models without the stopping rule that cut them short. The training script supports `--noise-ppm` together with `--patience` and `--min-delta`, so for example `--patience 1000 --min-delta 0` forces the full cosine schedule. Then rerun χ² scoring through `--denoise-template`.
3. Train the retrievals on noisy spectra with `--noise-ppm σ` (test noise follows by default), so the noisy-test R² values are no longer lower bounds.
4. Report random-forest results with `--train-copies 3` alongside the single-copy runs. The README quotes such a run, but no output file for it exists in the repository.

---

## W2. Statistical evidence is thin, and some comparisons are asymmetric or confounded

**Problem.**
- Every result uses a single train/validation/test split, and no AUC, R² or bias has a confidence interval.
- The design study uses one seed for the bottleneck sweep and two for most of the 2×2 ablation.
- The TauREx detector has one seed while the other two codes get the best of three. The text says this does not inflate TauREx's advantage, but nothing tests that.
- The width × schedule ablation changes the schedule and the early-stopping patience together, so the claimed "≈1.9× from the schedule" cannot be attributed.
- The AUC difference between neighbouring cells of the noise table (for example 0.669 vs 0.630) has no uncertainty attached.

**Fix using the codebase.**
1. Train the missing TauREx seeds: `anomaly.py --train-datasets taurex --init-seeds 43 44`, then `--eval-only`, which reapplies the same validation-based selection.
2. Separate schedule from patience with the two missing cells: `train.py --scheduler cosine --patience 20` and `--scheduler plateau --patience 50`, for both widths and seeds 42 and 43. This follows the pattern of the existing ablation launch script.
3. Repeat the full cross-code pipeline for 2–3 different split seeds. `--seed` sets the split in `anomaly.py`, `chi2_score.py`, `noise_robustness.py`, `rf_classifier.py` and `ml_retrieval.py`. Report mean ± std across splits.
4. Bootstrap the test planets for confidence intervals. The per-spectrum scores in `runs/anomaly/test_scores.npz` and the predictions in the retrieval `predictions.npz` files are already saved, and `anomaly.roc_auc` can be called inside a resampling loop.
5. Regenerate `runs/anomaly/chi2/` with the current scoring script. The committed file was written by an older version with a different schema.

---

## W3. The "retrieval" experiment uses an unconverged point estimator as a stand-in for Bayesian retrieval

**Problem.**
- The bias claims for Q3 come from XGBoost point estimates. Of the 27 regressors, all 15 for R⋆, Rp, Mp, T and CO₂, plus one of the three for H₂O, stopped at 2,767–2,999 of a 3,000-round cap. They are cap-limited, not converged.
- A median bias from an under-fitted regressor mixes code mismatch with emulator error.
- The learning-curve result that "more data widens the gap" may partly reflect the cap: larger subsets need more rounds to converge.
- Point estimates also cannot show whether a mismatched-code retrieval is overconfident, which is the question practitioners care about.

**Fix using the codebase.**
1. Rerun `ml_retrieval.py` with `--n-estimators 10000 --learning-rate 0.05`, check that every best round falls well below the cap, and confirm the self/cross gaps and the learning-curve trend.
2. Use the existing `--max-depth` and `--early-stopping` options for a small regularisation sweep. This addresses the overfitting already visible in the saved loss-curve figures.
3. For coverage, fit quantile regressors at the 16th, 50th and 84th percentiles. XGBoost supports this through its objective setting, which needs one change in the model-construction function. Report the fraction of cross-retrieval truths that fall inside the 16–84% interval for self vs cross.
4. Outside the codebase: run a nested-sampling retrieval with one code on a few hundred spectra generated by another, as an anchor for real inference.

---

## W4. The physical attributions are not supported, and the sample is heavily selected

**Problem.**
- The residual section reads the 1.55/2.31/2.36 µm features as correlated-k vs cross-section sampling, the POSEIDON−TauREx short-wavelength deficit as a continuum treatment, and the offset as a reference-pressure convention. None of these is tested.
- "Default opacities" means the codes may use different line lists, so opacity data, opacity method and code conventions are fully confounded.
- Separately, the 210 ppm cut removes 55% of planets and keeps 19% of planets below 600 K against 55% above 2000 K. The population statistics — offsets, the share of planets with ρ ≥ 30 ppm, the bias medians — are therefore statistics of a selected sample, and the paper never shows how they change with the cut.

**Fix using the codebase.**
1. Rerun the residual analysis on all 98,725 planets with `--no-cut`, and at 0, 100 and 210 ppm with `--min-feature-height-ppm`. Report how the offsets, the RMS residual and the component R² change.
2. Retrain detection and retrieval at a lower cut. `anomaly.py` already accepts `--min-feature-height-ppm`; the retrieval and classifier scripts need the same option, passed through to the `data.py` split function. Show which conclusions depend on the cut.
3. Test the "deterministic systematics" claim in practice. Use the random-forest component regressors already fitted in the residual analysis to predict and subtract the residual from POSEIDON spectra, then rerun cross-retrieval on the corrected spectra. If the CO and Rp gaps close, the emulation argument in the Conclusion is demonstrated rather than proposed.
4. For attribution (outside the codebase): regenerate a subset with all three codes on a shared line-list set, and with petitRADTRANS in line-by-line mode. Feed the result through the same residual pipeline. The difference between the two runs isolates the opacity-method term.

---

**Minor:** state in the main text that the autoencoder design study was done on POSEIDON-only data with a different planet set from the cross-code experiments.
