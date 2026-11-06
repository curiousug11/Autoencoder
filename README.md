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
2. Train neural-network emulators θ → S, validate their derivatives, and test whether the step-1 pairs are separate modes (below).
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

## Step 2: emulators and the mode test (`emulator.py`, `mode_paths.py`, `refine_modes.py`)

```
pip install -r requirements.txt                    # torch: CPU build is enough
python emulator.py --code pos --seeds 0 1 2 3      # ~35 min per member on one core; repeat for prt, taurex
python emulator.py --code pos --evaluate           # accuracy and Jacobian agreement -> ensemble_eval.json
python mode_paths.py --codes pos --threads 1       # one code per process; ~80 min per code
python mode_paths.py --noisy --out runs/mode_paths_noisy   # same with a noisy copy of each spectrum
python mode_paths.py --plots-only                  # merge per-code runs, redraw plots
python refine_modes.py                             # re-test every credible "separate" case (~1 h)
python check_false_minima.py                       # long descents of the poor-fit secondary minima
```

Step 1 found pairs of very different θ whose spectra are indistinguishable. That alone does not say whether they are two **separate modes** (two χ² minima with a barrier between them, the signature of a fold) or two points on one **continuous valley** (a sloppy direction). Telling the two apart needs χ²(θ) between the samples, so this step trains emulators and then searches for the lowest path between the two minima.

### Emulators

For each code there are 4 networks (seeds 0–3): an MLP with 4 hidden layers of 384 SiLU units, trained for 400 epochs on an 80/10/10 split that is the same for all three codes. The network predicts log y, the per-bin atmosphere height in units of Rp. The transit depth is then (0.10276·Rp/Rs)²(1+y)², so Rp/Rs, the stiffest direction, enters exactly and the network only has to learn the atmosphere. 26 POS spectra with a non-positive atmosphere height are left out. The ensemble mean is used for χ², and the members are used to check robustness.

Test-set error of the ensemble mean (RMS over the 52 bins, ppm; ~9,870 test spectra per code):

| code | median | 90% | median, feature-rich | 90%, feature-rich | spread between members (median) |
|---|---|---|---|---|---|
| POS | 0.24 | 3.1 | 1.11 | 9.1 | 0.19 |
| petitRADTRANS | 0.20 | 2.8 | 0.94 | 8.0 | 0.18 |
| TauREx | 0.25 | 3.6 | 1.19 | 9.8 | 0.21 |

- The typical error is 1–2% of the 30–50 ppm noise levels tested. The worst 1% (60–160 ppm) are spectra with very large features, where it is about 0.2% of the feature height. Before every test the run checks that the emulator reproduces both spectra of the pair to χ² < 2 at that σ. All 1,662–13,418 pairs per code and σ pass.
- **Derivatives**, which the descents and the path search use: on 1,000 test points, the four members' Jacobians agree to a median 0.02% (relative Frobenius norm; 90th percentile 0.2%). Column by column, the agreement is better than 0.6% for Rs, Rp, Mp and T, 2–7% for chem1, chem2 and chem4, and poorest for chem3 (12%) and chem5 (25–30%). Those two columns are also 10× smaller than the others: the spectra barely respond to those abundances, which matches chem3 being the sloppiest parameter in step 1.

### The mode test

For each pair (θ_i, θ_j), spectrum S_i is the data:
1. Descend χ² from θ_i and from θ_j to their local minima. This uses Levenberg–Marquardt in the unit box, with an active set for box faces and a trust region of 0.02 per step, so a descent cannot jump a barrier.
2. Find the lowest path between the two minima with the string method: 12 images, 120 iterations, endpoints free, and each image moved by the damped Gauss–Newton step perpendicular to the path.
3. The **barrier** is the highest χ² along the path, minus the χ² of the worse minimum. Results with a barrier below 0.5 (or minima < 0.02 apart) are **one basin**. A barrier ≥ 2 means **separate modes**. Separate modes are **credible** if both minima lie inside the 95% region (|χ²_a − χ²_b| < 16.9), i.e. the second mode would be a real alternative answer.

There are two controls, each run with the same machinery:
- **close control:** the second start is θ_i plus a random step of 0.05. These should come out as one basin, so the rate at which they don't measures false alarms.
- **random partner:** the second start is uniform over the prior box. This probes the whole χ² landscape around each data spectrum.

The test was run on 300 random emulator-faithful pairs per code and σ, plus 100 of each control. It was done on the noiseless spectra and again on a noisy copy (white noise at σ), for σ = 30 and 50 ppm.

An earlier version used Adam for both the descents and the path. Those runs invented barriers: 23% of close controls came out as "separate" when run longer. Levenberg–Marquardt with the trust region fixed it.

### Results

**No matched pair is two separate modes.** Of 1,800 matched pairs per run (300 for each of 3 codes and 2 σ):

| | noiseless | noisy |
|---|---|---|
| matched pairs, one basin | 1,800 (100%) | 1,786 (99.2%) |
| matched pairs, separate (first pass) | 0 | 10 (0–1% per code and σ), barriers Δχ² = 2.1–6.5; 8 credible |
| matched pairs, ambiguous (barrier 0.5–2) | 0 | 4 |
| close control, separate (first pass) | 0–4% | 0–6% |
| distance between the two minima (median) | 0.001–0.01 | 0.28–0.40 |
| χ² at the minima (median) | ~0 | 45–47 (expected: 52 − 9 = 43) |

- **Noiseless:** the descents from θ_i and θ_j end at the same point. The data fix one θ, and the step-1 partner simply lies in the valley that leads down to it. The step-1 degeneracies are therefore **continuous valleys**, not folds.
- **Noisy:** the two descents land on a valley floor that the noise has made slightly bumpy, typically 0.3 apart. The few first-pass "separate" pairs have barriers of only Δχ² = 2.1–6.5, at about the close control's false-alarm rate.

**Refinement of every credible separate case** (`refine_modes.py`, `refine.json`, `refine_barriers.png`). This covers 29 cases across both runs: 8 matched pairs, 8 close controls and 13 random partners. For each one the script:
- replays the exact data the first pass used, including the noise draw;
- re-descends both ends with fresh damping;
- runs the string method again with 24 images and 400 iterations from 16 starting routes: the straight line plus 15 bent through a random midpoint;
- keeps the lowest barrier.

**All 29 collapse.** 28 become one basin (barrier < 0.5, most < 0.02) and 1 is ambiguous (0.62). First-pass barriers of 718, 2,719 and 5,138 drop to zero. In those cases the first-pass endpoint was not a minimum but a point still sliding downhill. After refinement the ends either coincide or sit 0.5–1.7 apart on a flat valley floor.

**What the random-partner control shows.** In the first pass, 6–35% of random starts ended at a poor-fit second "minimum" (χ² in the thousands), most often for TauREx. The descent trace shows these descents had not stalled: every step was accepted at minimum damping and χ² was still falling. With the 0.02-per-step trust region, 300 steps are too few to cross the box along a curved valley. After 3×300 further steps, 93 of 115 reach a good fit (χ² < 16.9) after moving a median 1.2–1.4 across the box. Run out to 3,000 steps (`check_false_minima.py`), **110 of 115 reach a good fit**. 3 (TauREx, 30 ppm) are still descending. Only 2 stop at a poor fit: χ² = 68 (petitRADTRANS, 30 ppm) and χ² = 20.5 (TauREx, 50 ppm, just outside the 95% region). Genuine poor-fit local minima are therefore rare (≤ 2 in 600 random starts), and the high random-start "separate" rates in `summary.json` mostly measure descents that had not finished. This is also why the random-partner set is a weak positive control: the χ² landscape offers very few true barriers to find.

### What this means for the project

- In this 9-parameter, cloud-free model at 30–50 ppm, **the near-degeneracies are sloppy valleys**: the Rs–Rp–Mp scaling symmetry and the poorly constrained species (chem3, chem5) from step 1. No credible separate mode survived. That fits the expectation that a map from 9 dimensions into 52 is generically an immersion with no folds. Any multimodality has to come from the χ² landscape's focal structure (step 4), not from the surface folding onto itself.
- The valleys are long and curved, and χ² along them is flat to < 0.5. That makes them the natural object for step 3: they should appear as Fisher eigenvalues orders of magnitude below the rest, with eigenvectors that rotate along the valley.
- Starting far from the solution, the χ² landscape is hard to descend but almost always leads to the right valley. That is a practical warning for gradient-based retrievals with small step budgets: a stalled descent looks like a second mode.

**Caveats.**
- There are no cloud or haze parameters. Clouds are the usual source of multimodal retrievals in the literature, and they add a second way to flatten features.
- The test uses one noise realisation per case and 300 of the 1,662–13,418 pairs per code and σ.
- A path barrier is only an upper bound on the true pass. The refinement only re-tested the cases flagged as separate; for one-basin cases a better path could only lower the barrier further, so their classification stands.
- Everything depends on the emulators. They are accurate to ~1 ppm typical, and the matched-pair results agree across all four members and all three codes.

**Outputs.**
- `runs/emulator/{code}/`: `seed{0-3}.pt` (models), `seed*_metrics.json`, `ensemble_eval.json`.
- `runs/mode_paths/` and `runs/mode_paths_noisy/`:
  - `summary.json` and per-code `{code}_summary.json`;
  - `{code}_sigma{σ}.npz` (minima, barriers, per-member barriers, χ² along every path, and the matched-pair paths);
  - `refine.json`;
  - plots: `barrier_hist.png`, `minima_param_diffs.png`, `example_paths.png` (the largest first-pass barriers among matched pairs, before refinement).
- `runs/mode_paths/refine_barriers.png`: first-pass vs refined barrier for every case.
- `runs/mode_paths/false_minima_check.json`: the long descents.

## Step 3: the Fisher metric and its hierarchy (`fisher.py`)

```
python fisher.py              # ~4 min on 4 cores; writes runs/fisher/
python fisher.py --plots-only # redraw the plots
```

With white noise σ per bin, the Fisher information of θ is g/σ², where g = JᵀJ and J = ∂S/∂θ. g is the metric that the map θ → S induces on the 9-dimensional surface of spectra. It is evaluated with the emulator ensembles at 4,000 test-set planets, the same planets for every code. Two coordinate systems are used:
- **Log coordinates** (log₁₀ Rs, Rp, Mp, T, and the abundances, which are already in dex) for the hierarchy, as is usual for sloppy models. In these coordinates the step-1 directions are exact: Rp/Rs is (−1, 1, 0, …); the feature amplitude at fixed Rp/Rs is (½, ½, −1, 1, 0, …); scaling Rs, Rp and Mp together is (1, 1, 1, 0, …).
- **The unit prior box** for statements relative to the prior (all priors are uniform there) and for the widths.

**The eigenvalue hierarchy.** Values are log₁₀ λ in ppm² per dex², medians over planets. The three codes agree to 0.001–0.11 dex.

| | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|---|---|---|---|
| median | 11.1 | 6.0 | 4.6 | 3.7 | 2.9 | 2.3 | 1.6 | 0.8 | −0.1 |
| gap to next (decades) | 5.3 | 1.35 | 0.92 | 0.75 | 0.56 | 0.48 | 0.61 | 0.73 | |
| what it is (`eigenvectors.png`) | Rp/Rs | amplitude (T/Mp) | T and Mp | chem4, chem1, chem2 | chem4, chem1 | Rs·Rp·Mp scaling + chemistry | same | chem3, chem2, chem5 | chem5, chem3 |

- **The spectrum is sloppy, with one exceptionally stiff direction.** The eigenvalues span 11.4 decades at a typical planet. v1 is exactly Rp/Rs (overlap 1.000) and sits 5 decades above the rest. Below it, the other eight fall roughly geometrically, 0.5–1.4 decades apart, over 6 decades. That is the pattern sloppy models across science show (Gutenkunst et al. 2007; Transtrum, Machta & Sethna 2010, 2011). From planet to planet each eigenvalue varies by about ±2 dex (68%), almost entirely because of the feature height (next bullet).
- **Feature height sets the overall scale** (`eigenvalues_vs_features.png`). Every eigenvalue below v1 grows as (feature height)², and v1 as about its first power. So the whole ladder slides down together towards featureless spectra, and the featureless limit from step 1 is the boundary where v2–v9 all shrink to zero together.
- **The step-1 directions, tested:**
  - Rp/Rs is v1.
  - The feature amplitude at fixed Rp/Rs lies 98% in v2, which is mostly T/Mp, as expected for the scale height.
  - v3 is a second Mp–T combination, so T and Mp are each partly constrained, not only their ratio.
  - **Scaling Rs, Rp and Mp together is not the sloppiest direction.** It falls in v4–v7, with its Fisher eigenvalue about 10⁴·⁴ ppm² per dex², which gives a 1σ width of about 0.2 dex at 30 ppm. Pressure-dependent opacity and the chord geometry (√(Rp·H)) break the symmetry. It is a weak direction, not a near-exact one.
  - The truly sloppy directions are chemical: chem3 and chem5, mixed with chem2.

**How many directions the data constrain** (`constrained_directions.png`). The effective number of constrained directions relative to the uniform prior is N_eff = Σ λ_k/(λ_k + 12σ²) in the unit box. Its median is 5.8, 4.6, 3.9 and 2.9 at σ = 10, 30, 50 and 100 ppm. At 30 ppm it ranges from 1.3 to 6.9 across planets (16–84%), depending on feature height.

Gaussian marginal posterior widths at 30 ppm, as a fraction of the prior width:

| Rs | Rp | Mp | T | chem1 | chem2 | chem3 | chem4 | chem5 |
|---|---|---|---|---|---|---|---|---|
| 0.77 | 0.46 | 0.64 | 0.40 | 0.80 | 0.84 | 0.98 | 0.57 | 0.98 |

chem3 and chem5 are essentially unconstrained, and chem4 and T are the best constrained.

**A hyper-ribbon** (`widths.png`). The width of the surface along each unit-box eigendirection is the arc length in data space of the straight line through the planet, from box face to box face. Where the emulator would leave its training range, the line stops there instead. That happens towards corners with tiny Mp and large Rp and T, for 0–28% of lines, mostly along v2. Median widths in ppm:

| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|---|---|---|
| 310,000 | 3,500 | 940 | 490 | 265 | 135 | 78 | 50–59 | 64–72 |

- From v2 to v8 each width is about 0.55 times the previous one: the surface is a long, thin ribbon, as sloppy-model theory predicts (Transtrum et al. 2011).
- For a typical planet 7 of the 9 widths exceed 2σ at 30–50 ppm. Over the full prior range, then, the data can see 7 dimensions, but at the precision of a single planet only about 4.6 are constrained (N_eff).
- **v9 breaks the series.** The locally sloppiest direction is not the globally narrowest, and its arc is 21 times longer than the straight-line estimate √λ × (parameter span). Along v6–v9 the spectrum's sensitivity grows rapidly away from the planet. That is strong nonlinearity, whether it bends the surface or only stretches its parametrisation, and step 4 tells the two apart.

**Robustness.**
- Eigenvalues from individual ensemble members agree with the ensemble mean to a median of ≤ 0.02 dex for v1–v7, 0.05 dex for v8 and 0.14 dex for v9.
- The eigen-subspaces of different codes differ by angles with sine ≤ 0.13.

**Outputs** in `runs/fisher/`:
- `summary.json`;
- `{code}_fisher.npz` (per-planet eigenvalues and eigenvectors in both coordinate systems, and the widths);
- plots: `eigenvalues.png`, `eigenvectors.png`, `constrained_directions.png`, `eigenvalues_vs_features.png`, `widths.png`.

## Step 4: curvature and the focal set (`curvature.py`)

```
python curvature.py              # after fisher.py; ~30 min on 4 cores; writes runs/curvature/
python curvature.py --plots-only # redraw the plots
```

**What is computed.** This uses 2,000 of the step-3 planets, the same ones for every code. The emulator Hessians H = ∂²S/∂u² are exact autograd second derivatives; they match finite differences to 10⁻⁹. Their part normal to the surface is the second fundamental form. In the Fisher eigenbasis, scaled so that a unit step moves the spectrum by 1 ppm, it becomes 52 symmetric 9×9 matrices B_k (units 1/ppm). From these:
- **Normal curvature** along a unit direction u is the data-space vector B(u,u). The largest principal curvature is κ_max = max |B(u,u)|. Its inverse ρ_c is the **distance from the surface to its nearest focal point**.
- **Focal set.** Data y = S(θ) + ε with ε normal to the surface leave θ a critical point of χ², with Hessian I − Σ ε_k B_k in this frame. The minimum splits (a fold or cusp of the distance-squared function) exactly when λ_max(Σ ε_k B_k) > 1, i.e. when the data cross the focal set. For white noise, P_focal(σ) is the probability that noise carries the data across it near the true planet (Monte Carlo, 512 draws per planet).
- **Bates & Watts (1980) intrinsic nonlinearity** γ = κ_max·σ·√9. The Gaussian (Fisher) approximation is adequate when γ < 0.34, i.e. when ρ_c > 8.7σ.
- **Intrinsic curvature** via the Gauss equation, which is exact in flat data space: sectional curvature K(a,b) = B_aa·B_bb − |B_ab|², and the scalar curvature.
- Each eigendirection k gets a **curvature radius** ρ_k = 1/|B_kk|. It is compared with the step-3 width W_k through the **sagitta** W_k²/(8ρ_k), the distance by which the surface departs from straight across its full width.
- **"Data-constrained" variants** use only the eigendirections the data constrain better than the prior (λ_k > 12σ² in the unit box). That is a median of 5 directions at 30 ppm and 4 at 50 ppm.

**Reliability.** Second derivatives are where an emulator is weakest, and whitening amplifies their errors along sloppy directions. Every quantity is therefore computed for each ensemble member too, and a planet counts as reliable for a quantity only if the 4 members agree within a factor of 1.5. The members agree well on the curvature along v2–v5 (reliable at 74–95% of planets) and poorly along v8–v9 (13–22%). So κ_max over all directions, which is dominated by v9, is reliable at only 17% of planets; the data-constrained κ_max is reliable at 67–78%. Among reliable planets, the codes agree on the data-constrained ρ_c to a median of 0.03–0.04 dex.

### Results (numbers are ranges over the three codes)

**1. The surface is strongly curved: it is a curled ribbon, not a flat one** (`radius_vs_width.png`).

| | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|---|---|---|---|
| curvature radius ρ_k (ppm, median) | ~2×10⁹ | 1,400 | 290 | 190–210 | 145–155 | 80 | 35 | 11–15 | 9–13 |
| width W_k (ppm, median) | 3×10⁵ | 3,500 | 960 | 500 | 285 | 142 | 76 | 50–62 | 64–71 |
| sagitta W²/8ρ (ppm, median) | 6 | 900 | 340 | 140–180 | 75 | 42 | 21–38 | 10–33 | 19–21 |
| planets with sagitta > 30 ppm | 30% | 80% | 73% | 67% | 60% | 53% | 46–53% | 41–50% | 44–46% |

- Along Rp/Rs (v1) the surface is flat.
- Along every other direction the radius of curvature is only 0.25–0.5 of the width. Across its full extent the surface turns through 2–4 radians, and the radius shrinks in step with the width. So the hyper-ribbon from step 3 is a geometric series in curvature as well as in width, as Transtrum et al. (2011) describe for sloppy models.
- It is not "low curvature" relative to the widths, though, unlike the examples in Transtrum et al. (2010). The sagitta exceeds 30 ppm at roughly half the planets even along the sloppiest directions, so over the prior range the bending is visible at realistic noise.
- Physically, each abundance and T moves a feature through appearance, growth and saturation, which traces a strongly bent curve in data space.

**2. Along the sloppy directions the focal set lies almost on the surface, but crossing it changes nothing** (`focal_distance.png`, dashed).
- The most curved direction is v9 at 86–89% weight (v8 at 5%). ρ_c has a median of 1.1–1.3 ppm over all planets (5–95%: 0.005–100 ppm), or 5–7 ppm over reliable planets. So at realistic noise most planets' data lie beyond the focal set: P_focal = 59–63% on average at 30 ppm, and 42–47% even at 10 ppm.
- The validation (`focal_validation.png`, top row) shows such crossings do split the minimum, but the two new minima differ from the old critical point by Δχ² ≤ 0.014 at 30 ppm. The surface is so thin in those directions that the bending is invisible. This is the flat-valley behaviour seen in step 2.

**3. Along the directions the data constrain, the focal set is about 5σ away and noise rarely crosses it** (`focal_distance.png`, solid; `focal_probability.png`).
- At σ = 30 ppm the data-constrained ρ_c has a median of 160–175 ppm (5%: 56 ppm; 16%: 80 ppm). The median ρ_c/σ is 5.1–5.8 at both 30 and 50 ppm, because fewer directions count as constrained at the higher noise.
- It depends on feature height: the median ρ_c is 92–99 ppm for features of 30–210 ppm, 144 ppm for 210–500 ppm, and 275–500 ppm for features above 500 ppm. Below 30 ppm only Rp/Rs is constrained, and it is flat.
- The probability that noise crosses this part of the focal set is small. P_focal averages 0.6–0.7% over planets at both 30 and 50 ppm; 11–13% of planets have P > 1% and 3.7–4.7% have P > 5%. The highest values are for features of 30–210 ppm (mean P ≈ 1–1.5%) and for the very largest features above 3,000 ppm (4–6%, but only about 25 planets, where the emulator is also least accurate).
- **The Gaussian approximation still often fails.** By the Bates–Watts criterion the posterior is noticeably non-Gaussian at 60–66% of planets. The curvature bends the posterior into a banana far more often than it splits it.

**4. When the data do cross the focal set, a genuine split appears, but with a small barrier** (`focal_validation.png`, bottom row).
- Data are placed at t·ρ_c along the most curved data-constrained normal. The exact χ² Hessian follows the predicted 1 − t to within 5×10⁻⁴ in every case. The minimum stays put for t < 1 and splits for t > 1.
- For the most curved planets (ρ_c ≈ σ), the barrier between the two new minima is Δχ² ≈ 0.1 at t = 1.1, 0.5–0.6 at t = 1.5 and 1.3–3.2 at t = 2.5, at 30 ppm. The new minima often run to a prior box face.
- For typical planets (ρ_c ≈ 160 ppm), data that far off the surface (≥ 5σ) already destabilise the sloppy directions before the constrained one. Δχ² reaches 2–3 at t ≈ 0.9 and 55–73 at t = 2.5 (13σ), but noise that large is not realistic.
- So at realistic noise a focal crossing broadens and flattens the χ² basin instead of creating a separate mode. That is what step 2 measured directly: no barrier ≥ 0.5 survived refinement in 1,800 noisy pairs per run.

**5. Intrinsic curvature.** In the stiffest plane (v1–v2), the sectional curvature radius has a median of 4.5–6.9×10⁶ ppm, measurable at 40–57% of planets, so the surface is intrinsically flat where the data are most informative. In the sloppiest plane (v8–v9) the radius is about 30–46 ppm, comparable to the noise, and mostly negative (saddle-like; 76–87% of cases). But the ensemble members agree there at only 5–6.5% of planets. The scalar curvature, which those planes dominate, agrees between members at only about 2% of planets, so it is reported but not interpreted. Intrinsic curvature therefore matters only along the sloppy chemical directions, where these emulators cannot measure it reliably.

### What steps 3 and 4 say about the project question

- **Eigenvalue hierarchy.** The model manifold is sloppy. Rp/Rs alone is 5 decades stiffer than anything else. The other 8 directions fall geometrically, about 0.5–1.4 decades apart, over 6 decades. All of them scale with (feature height)², and the featureless edge is where they vanish together. Among the step-1 hypotheses, Rp/Rs and the amplitude are confirmed as the stiff directions. The Rs–Rp–Mp scaling is weak but not the sloppiest; that place belongs to chem3, chem5 and chem2.
- **Curvature.** Strong. Extrinsic curvature radii are 0.25–0.5 of the widths along every direction except Rp/Rs, so the surface is a curled hyper-ribbon. Intrinsic curvature is negligible in the stiff plane; in the sloppy planes it cannot be pinned down with these emulators.
- **Boundaries.** The prior box faces cut the ribbon. When minima split they usually run to a face. The featureless limit is a boundary where 8 directions collapse together.
- **Folds and caustics.** The focal set exists and sits very close to the surface along the sloppy directions, so noise crosses it routinely, but those crossings produce only invisible ripples. Along the constrained directions it lies about 5σ out, and crossing it has a probability below about 1% per planet at 30–50 ppm. Even then it creates only shallow splits unless the data are far beyond it. **For this cloud-free, 9-parameter model, focal-set crossings are real but are not a practical source of multimodal posteriors.** The geometry predicts non-Gaussian, banana-shaped posteriors at most planets instead. Step 5 can test both predictions with full posterior sampling on the emulators, including whether planets with high P_focal, mostly features of 30–210 ppm, show more multimodality.

**Caveats.**
- All second-derivative results come from emulators and are trusted only where the 4 members agree. Curvature along v8–v9 and the scalar curvature are mostly not measurable this way. A finite-difference check with the radiative-transfer codes themselves at a few planets would settle them.
- P_focal is evaluated at the true planet. That is the local criterion; it does not count distant secondary minima, which step 2 searched for directly.
- There are no cloud parameters. Clouds add a degenerate, strongly nonlinear direction, which is the most likely way to bring the focal set closer.

**Outputs** in `runs/curvature/`:
- `summary.json` (including the validation rows);
- `{code}_curvature.npz`: per planet, κ_max and its tangent and normal directions; the data-constrained versions; normal curvature per direction; sectional and scalar curvature; the focal-crossing samples λ_max(Σ z_k B_k); and the member-agreement ratios;
- plots: `focal_distance.png`, `radius_vs_width.png`, `focal_probability.png`, `focal_validation.png`.

### References
- Bates, D. M., & Watts, D. G. (1980). Relative curvature measures of nonlinearity. *Journal of the Royal Statistical Society B*, 42(1), 1–25. https://doi.org/10.1111/j.2517-6161.1980.tb01094.x
- Gutenkunst, R. N., et al. (2007). Universally sloppy parameter sensitivities in systems biology models. *PLoS Computational Biology*, 3(10), e189. https://doi.org/10.1371/journal.pcbi.0030189
- Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2010). Why are nonlinear fits to data so challenging? *Physical Review Letters*, 104, 060201. https://doi.org/10.1103/physrevlett.104.060201
- Transtrum, M. K., Machta, B. B., & Sethna, J. P. (2011). Geometry of nonlinear least squares with applications to sloppy models and optimization. *Physical Review E*, 83, 036701. https://doi.org/10.1103/physreve.83.036701
