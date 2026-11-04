# Spectrum autoencoder — datasets

Data-only branch. Contains just the `data/` files used by the code on the main branches.

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

# Project: geometry of the spectral manifold

The 9 parameters θ map through radiative transfer to a 52-bin spectrum S(θ), so the spectra form a 9-dimensional surface in 52-dimensional data space. The goal is to measure that surface's Fisher-metric eigenvalue hierarchy ("sloppiness"), curvature and boundaries, and to test whether multimodal retrieval posteriors come from its geometry. Two mechanisms can produce them:

1. **Global:** the surface passes close to itself, so distant θ give nearly the same spectrum.
2. **Local:** the noise is comparable to the surface's curvature radius, so the data lie near its focal set, where the number of χ² minima changes.

Plan:
1. Model-free scan of the samples for close approaches (`close_pairs.py`, below).
2. Train neural-network emulators θ → S and validate their derivatives.
3. Map the Fisher eigenvalue hierarchy across the prior for all three codes.
4. Compute curvature and the focal set.
5. Test multimodal posteriors against the geometry.

## Step 1: close approaches in the samples (`close_pairs.py`)

```
pip install -r requirements.txt
python close_pairs.py            # ~6 min on 4 cores; writes runs/close_pairs/
```

Two spectra count as **indistinguishable** at white noise σ per bin when ‖S_i − S_j‖²/σ² < χ²₉₅(9) = 16.92, i.e. θ_j would sit inside the 95% region of a 9-parameter posterior if S_i were observed. The scan uses the absolute spectra (transit depth), not the per-spectrum normalised ones from the autoencoder work. Feature-rich means max − min ≥ 210 ppm, the same cut as before (about 44k of 98.7k spectra). Parameter distances are measured in the unit prior box, where two random spectra are a median of 1.13 apart.

**How often a spectrum has an indistinguishable partner** (POS; petitRADTRANS and TauREx agree to within 0.01):

| σ (ppm) | featureless (< 210 ppm) | feature-rich | feature-rich pairs | parameter distance, median (16–84%) |
|---|---|---|---|---|
| 10 | 0.63 | 0.001 | 17 | 0.53 (0.30–0.79) |
| 20 | 0.83 | 0.016 | 333 | 0.58 (0.38–0.83) |
| 30 | 0.93 | 0.069 | 1,798 | 0.63 (0.41–0.89) |
| 50 | 0.98 | 0.22 | 13,418 | 0.72 (0.49–0.99) |

- **Featureless spectra collapse.** When the scale height is small, the spectrum tends to a flat line set only by (Rp/Rs)². That is a boundary of the manifold where 8 of the 9 directions vanish, and almost every featureless spectrum has a partner from very different parameters.
- **Feature-rich spectra rarely match by chance, but when they do the parameters are far apart.** Matched pairs are typically 0.5–0.7 apart, more than half the separation of two random spectra.

**What stays pinned** (σ = 30 ppm; median |Δ| in matched pairs relative to random feature-rich pairs; all three codes agree):

| Rs | Rp | Mp | T | chem1 | chem2 | chem3 | chem4 | chem5 | Rp/Rs | T·Rp³/(Mp·Rs²) | mean log X |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.72 | 0.71 | 0.50 | 0.40 | 0.38 | 0.41 | 0.85 | 0.31 | 0.57 | **0.004** | **0.12** | 0.59 |

- **Stiffest direction:** Rp/Rs, pinned to 0.001 dex.
- **Next:** T·Rp³/(Mp·Rs²), which is proportional to the feature amplitude 2·Rp·H/Rs² with H ∝ T·Rp²/Mp. Scaling (Rs, Rp, Mp) together by the same factor leaves both combinations unchanged, so Rs, Rp and Mp individually slide almost freely. That is an approximate continuous symmetry, and it should show up as a near-zero Fisher eigenvalue in step 3.
- **Sloppiest species:** chem3 is barely constrained (0.85). chem4 and chem1 are the best constrained.

**Species swaps.** In about 4% of matched feature-rich pairs at 30 ppm (9% at 50 ppm), one abundance rises by ≥ 2 dex while another drops by ≥ 2 dex. The most common combinations are chem1↔chem3, chem3↔chem4 and chem2↔chem3, with almost the same counts in all three codes (`species_swap_counts` in `summary.json`). These are the leading candidates for genuinely separate modes, where one molecule substitutes for another. Because chem3 is so weakly constrained, some of them may instead be chem3 drifting freely. `distant_pair_examples.png` shows the most distant matched pair per code.

**Separate basin or one continuous valley? The samples alone can't tell.** The sampling is sparse: the median nearest-neighbour distance is 0.23 of the prior range. For a feature-rich spectrum, the cheapest move to one of its 15 nearest parameter-space neighbours costs a median Δχ² of 17,000 at 30 ppm (6,300 at 50 ppm). The threshold is 16.9, and fewer than 2% of spectra have any neighbour under it (`sampling_resolution.png`).

So a chain of samples can't follow a valley. I tried a minimax-path ("mountain pass") barrier on the neighbour graph, and it flagged almost every pair as separated, including pairs inside obviously continuous degeneracies, so it was dropped. Deciding between one basin and two needs the emulator in step 2: evaluate χ² along the path between each pair and look for a barrier.

**Outputs** in `runs/close_pairs/`:
- `summary.json`: all numbers above, per code and σ.
- `{code}_pairs.npz`: every feature-rich matched pair up to 50 ppm (indices `i`, `j`, `sigma_min_ppm`, the smallest σ at which the pair is indistinguishable, and `param_dist`). These are the input for step 2.
- `{code}_pairs_top.csv`: the 500 most indistinguishable pairs with their parameters.
- Plots: `partner_fraction.png`, `param_spread.png`, `sampling_resolution.png`, `distant_pair_examples.png`.
