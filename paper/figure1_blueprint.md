# Figure 1 blueprint: study pipeline

**Format:** full-width two-column figure (`figure*`), about 17.8 cm × 7.5 cm, vector PDF, sans-serif text at 7–8 pt. It is read left to right in five panels, (a)–(e), with panels (c)–(e) stacked as three horizontal lanes on the right.

**Colour code:** one fixed colour per code, used in every figure of the paper.
- POSEIDON: blue (#1f77b4)
- petitRADTRANS: orange (#ff7f0e)
- TauREx: green (#2ca02c)

Grey (#7f7f7f) marks processing steps. Black dashed lines mark the oracle and other references.

## (a) Simulation (left, about 25% of the width)

- **Top:** a box labelled $\boldsymbol\theta \in \mathbb{R}^9$, listing $R_\star, R_p, M_p, T$ and $\log X$ for H₂O, CH₄, CO, CO₂ and NH₃. Below it, a small inset of the $T$ vs $\log X_{\rm CH_4}$ prior showing the cap above 2000 K, with an equivalent NH₃ cap at 1600 K.
- **Middle:** three arrows fan out to three code boxes, each in its code colour:
  - "POSEIDON — cross-sections, R = 15,000"
  - "petitRADTRANS — correlated-k, R = 1,000"
  - "TauREx — cross-sections, R = 15,000"
- **Shared footer** under the three boxes: "isothermal · cloud-free · default opacities".
- **Bottom:** the arrows converge into a "bin to 52 Ariel bins (0.55–7.28 µm)" box. Next to it, a real example planet drawn as its three overlaid spectra: transit depth in ppm vs log λ. Use a planet with large inter-code differences (ρ ≳ 50 ppm) so the curves separate visibly.
- **Annotation:** "N = 98,725 planets × 3 codes; same θ in every code".

## (b) Preprocessing (narrow column, about 12%)

Three grey boxes stacked vertically, joined by arrows:
1. "Feature-height cut h ≥ 210 ppm (all codes) → 43,885 planets"
2. "Optional white noise ε ~ N(0, σ²I), σ ∈ {2, …, 100} ppm"
3. "Per-spectrum standardisation ỹ = (y − m)/s; keep (m, s)"

Below them, a split bar divided 70/15/15 and labelled "train / val / test (by planet, shared across codes)".

## (c) Q1, structure of systematics (top lane)

- **Left:** the three-spectrum example from (a) becomes the difference spectrum Δ^(ab) (black), split into a horizontal dashed offset line δ and the residual shape r.
- **Middle:** a box "PCA of r / h̄ per code pair", with a thumbnail of the first 2 principal-component loadings vs wavelength. Mark the 1.55 and 2.35 µm features.
- **Right:** a box "RF regression of the PC scores on θ → out-of-sample R²".

## (d) Q2, detectability (middle lane, the tallest)

Two sub-blocks side by side.

**d1. One-class autoencoders.** Draw the AE-W layer stack as trapezoids with layer widths proportional to log(width):
- encoder 52 → 256 → 128 → 64 → 32 → 16 → **8**
- decoder 8 → 16 → 32 → 64 → 128 → 256 → 52
- the latent layer as a narrow bar labelled "z ∈ ℝ⁸ (linear)", and the output as "linear"

Draw three stacked copies in the three code colours, labelled "one AE per code". From the output, two score branches:
- e_c (z-space MSE), threshold τ^(95/99) from validation
- χ²_ν against the error bars, with τ^th = 1.512 and τ^emp

A grey "denoising variant" tag hangs from the AE stack: noisy input → clean target.

**d2. Three-class random forest.** A tree icon labelled "RF, 300 trees", mapping ỹ^σ to (p_POS, p_pRT, p_TRX). The pairwise score is p_b / (p_a + p_b).

A dashed black box labelled "oracle: χ² vs the true spectrum of code c (upper bound)" sits under both sub-blocks. A shared arrow leads to the output "AUC / TPR at 1% FPR vs σ".

## (e) Q3, retrieval bias (bottom lane)

- **Left:** a box "XGBoost, 9 regressors per training code", with input "[ỹ, log m, log s] ∈ ℝ⁵⁴".
- **Right:** a 3 × 3 grid with rows labelled "trained on" and columns "tested on", in the code colours.
  - Shade the diagonal light grey and label it "self".
  - Off-diagonal cells are labelled "cross".
  - One example cell holds a small bar giving the bias β and half-width w.
- **Output arrow:** "R², median bias β, |β|/w vs σ".

## Typography and checks

- Every symbol in the figure must match the notation in Sections 3–4 (θ, ỹ, Δ^(ab), δ, r, ρ, e_c, χ²_ν, β, w).
- No software variable names or file paths appear in the figure.
- Panel labels (a)–(e) go in bold in the top-left corner of each panel.
- The figure must read in greyscale: also give each code its own line style (solid, dashed, dotted).
