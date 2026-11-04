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

**What the random-partner control shows.** In the first pass, 6–35% of random starts ended at a poor-fit second "minimum" (χ² in the thousands), most often for TauREx. The descent trace shows these descents had not stalled: every step was accepted at minimum damping and χ² was still falling. With the 0.02-per-step trust region, 300 steps are too few to cross the box along a curved valley. After 3×300 further steps, 93 of 115 reach a good fit (χ² < 16.9) after moving a median 1.2–1.4 across the box. FALSE_MINIMA_RESULT

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
