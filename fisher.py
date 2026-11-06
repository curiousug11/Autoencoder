"""Step 3: the Fisher metric of the spectral manifold, its eigenvalue hierarchy and its widths.

For white noise sigma per bin, the Fisher information of theta is g / sigma^2 with g = J^T J and
J = dS/dtheta (ppm per unit coordinate). g is the metric that the map theta -> S induces on the
9-dimensional surface of spectra in 52-dimensional data space. This script evaluates it with the
emulator ensembles at the same test-set planets for all three codes and reports:

- the eigenvalue hierarchy ("sloppiness") in log coordinates, and how many decades it spans;
- what the eigenvectors are, compared with the step-1 directions (Rp/Rs, the feature amplitude
  T Rp^3 / (Mp Rs^2), and scaling Rs, Rp and Mp together);
- prior-relative constraints in the unit box (uniform priors): the effective number of
  constrained directions N_eff(sigma) = sum_k lam_k / (lam_k + 12 sigma^2) and each parameter's
  Gaussian posterior width as a fraction of its prior width;
- how the eigenvalues change with feature height (the featureless boundary);
- the widths of the surface along each eigendirection: the arc length in data space of the
  straight line through the planet along that direction, from box face to box face. Sloppy
  models are typically "hyper-ribbons" whose widths fall off geometrically;
- agreement between ensemble members and between the three codes.

    python fisher.py                    # ~10 min on 4 cores; writes runs/fisher/
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry import (CODE_COLOR, CODE_LABEL, CODES, DIRECTIONS, PARAM_NAMES, box, derivatives,
                      eigh_desc, eval_points, frozen_ensemble, load_data, log_factors, subspace_sin)

PRIOR_PRECISION = 12.0  # 1 / variance of a uniform distribution on [0, 1]


def pct(x, axis=0):
    return np.percentile(x, [16, 50, 84], axis=axis)


@torch.no_grad()
def widths(models, theta, lo, hi, V, n_line=33, chunk=200):
    """Arc length (ppm) of S along the line through each point in each unit-box eigendirection,
    clipped to the box. V: [n, 9, 9] eigenvectors in columns. Returns arc [n, 9], chord [n, 9], span [n, 9]."""
    lo_t, span_t = torch.as_tensor(lo), torch.as_tensor(hi - lo)
    u0 = (theta - lo) / (hi - lo)
    arc, chord, tspan = [], [], []
    for a in range(0, len(u0), chunk):
        u, v = u0[a:a + chunk, :, None], V[a:a + chunk]  # [b, 9, 1], [b, 9(coord), 9(k)]
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(v > 0, (1 - u) / v, np.where(v < 0, -u / v, np.inf)).min(1)  # [b, 9(k)]
            dn = np.where(v > 0, -u / v, np.where(v < 0, (1 - u) / v, -np.inf)).max(1)
        t = dn[..., None] + (up - dn)[..., None] * np.linspace(0, 1, n_line)  # [b, k, L]
        pts = u[:, None, :, 0][:, :, None, :] + t[..., None] * np.swapaxes(v, 1, 2)[:, :, None, :]
        pts = np.clip(pts, 0, 1)
        th = torch.as_tensor(lo + pts.reshape(-1, 9) * (hi - lo))
        s = torch.stack([m(th) for m in models]).mean(0).numpy().reshape(*pts.shape[:3], -1) * 1e6
        seg = np.linalg.norm(np.diff(s, axis=2), axis=-1)
        arc.append(seg.sum(-1))
        chord.append(np.linalg.norm(s[:, :, -1] - s[:, :, 0], axis=-1))
        tspan.append(up - dn)
    return np.concatenate(arc), np.concatenate(chord), np.concatenate(tspan)


def analyse(code, idx, lo, hi, args):
    t0 = time.time()
    models = frozen_ensemble(code, args.emulator_dir)
    S, P, _, _ = load_data(code, args.data_dir)
    theta = P[idx]
    Sm, Jm = derivatives(models, theta, lo, hi)  # [M, n, 52], [M, n, 52, 9]
    J = Jm.mean(0)
    c = log_factors(theta, lo, hi)
    Jl, Jl_m = J * c[:, None, :], Jm * c[None, :, None, :]
    lam_log, V_log = eigh_desc(np.swapaxes(Jl, 1, 2) @ Jl)
    lam_log_m = eigh_desc(np.swapaxes(Jl_m, -1, -2) @ Jl_m)[0]
    lam_u, V_u = eigh_desc(np.swapaxes(J, 1, 2) @ J)
    V_u_m = eigh_desc(np.swapaxes(Jm, -1, -2) @ Jm)[1]
    feat = S[idx].max(1) - S[idx].min(1)
    print(f"[{code}] derivatives {time.time() - t0:.0f}s", flush=True)
    arc, chord, tspan = widths(models, theta, lo, hi, V_u)
    print(f"[{code}] widths {time.time() - t0:.0f}s", flush=True)
    return dict(theta=theta, feat_ppm=feat * 1e6, lam_log=lam_log, V_log=V_log, lam_log_members=lam_log_m,
                lam_u=lam_u, V_u=V_u, V_u_members=V_u_m, width_arc=arc, width_chord=chord, width_tspan=tspan)


def summarise(r, args):
    lam, V = r["lam_log"], r["V_log"]
    L = np.log10(np.maximum(lam, 1e-300))
    out = {"n_points": int(len(lam))}
    q = pct(L)
    out["log10_eigenvalues_log_coords"] = {"p16": q[0].round(3).tolist(), "median": q[1].round(3).tolist(),
                                           "p84": q[2].round(3).tolist(),
                                           "units": "ppm^2 per (dex)^2; 1-sigma width along v_k is sigma/sqrt(lam_k) dex"}
    out["decades_spanned_median"] = float(np.median(L[:, 0] - L[:, -1]))
    out["consecutive_log10_gap_median"] = np.median(-np.diff(L, axis=1), 0).round(3).tolist()
    # eigenvector composition: median squared component of parameter j in eigenvector k
    comp = np.median(V**2, axis=0).T  # [k, j]
    out["eigenvector_composition_median_sq"] = {f"v{k + 1}": dict(zip(PARAM_NAMES, comp[k].round(3).tolist()))
                                                for k in range(9)}
    dirs = {}
    for name, d in DIRECTIONS.items():
        ov = np.einsum("j,njk->nk", d, V) ** 2  # overlap of d with each eigenvector
        rq = np.einsum("j,njk,nk,nik,i->n", d, V, lam, V, d)  # d^T F d
        cum = np.cumsum(ov, 1)
        dirs[name] = {
            "overlap_sq_with_v_k_median": np.median(ov, 0).round(3).tolist(),
            "fraction_in_stiffest_m_median": np.median(cum, 0).round(3).tolist(),
            "log10_rayleigh_over_lam1_median": float(np.median(np.log10(rq / lam[:, 0]))),
            "log10_rayleigh_over_lam9_median": float(np.median(np.log10(rq / np.maximum(lam[:, -1], 1e-300)))),
        }
    out["step1_directions"] = dirs
    # prior-relative constraints (unit box, uniform priors)
    lu = r["lam_u"]
    out["N_eff"] = {}
    out["posterior_width_over_prior_width"] = {}
    for s in args.sigmas:
        ne = (lu / (lu + PRIOR_PRECISION * s**2)).sum(1)
        out["N_eff"][f"{s:g}"] = pct(ne).round(3).tolist()
        Vu = r["V_u"]
        post = np.einsum("njk,nk,nik->nji", Vu, 1 / (lu / s**2 + PRIOR_PRECISION), Vu)
        w = np.sqrt(np.einsum("njj->nj", post) * PRIOR_PRECISION)
        out["posterior_width_over_prior_width"][f"{s:g}"] = dict(zip(PARAM_NAMES, np.median(w, 0).round(3).tolist()))
    # feature dependence
    fb = np.array(args.feature_bins)
    k = np.digitize(r["feat_ppm"], fb)
    out["log10_eigenvalues_by_feature_height"] = {
        f"{fb[i - 1]:g}-{fb[i]:g} ppm" if 0 < i < len(fb) else (f"<{fb[0]:g} ppm" if i == 0 else f">{fb[-1]:g} ppm"):
        {"n": int((k == i).sum()), "median": np.median(L[k == i], 0).round(2).tolist()}
        for i in range(len(fb) + 1) if (k == i).sum() >= 20}
    # widths (unit-box eigendirections)
    out["width_arc_ppm"] = {"p16": pct(r["width_arc"])[0].round(2).tolist(),
                            "median": np.median(r["width_arc"], 0).round(2).tolist(),
                            "p84": pct(r["width_arc"])[2].round(2).tolist()}
    lin = np.sqrt(lu) * r["width_tspan"]
    out["width_arc_over_linear_median"] = np.median(r["width_arc"] / lin, 0).round(3).tolist()
    out["n_widths_above_2sigma_median"] = {f"{s:g}": float(np.median((r["width_arc"] > 2 * s).sum(1)))
                                           for s in args.sigmas}
    # robustness: members vs ensemble mean
    Lm = np.log10(np.maximum(r["lam_log_members"], 1e-300))
    out["member_abs_dlog10_lam_median"] = np.median(np.abs(Lm - L[None]), (0, 1)).round(4).tolist()
    out["member_subspace_sin_median"] = [float(np.median(subspace_sin(r["V_u"][None], r["V_u_members"], m)))
                                         for m in range(1, 9)]
    return out


def compare_codes(res):
    out = {}
    for a, b in (("pos", "prt"), ("pos", "taurex"), ("prt", "taurex")):
        if a in res and b in res:
            La, Lb = (np.log10(np.maximum(res[c]["lam_log"], 1e-300)) for c in (a, b))
            out[f"{a}-{b}"] = {
                "abs_dlog10_lam_median": np.median(np.abs(La - Lb), 0).round(3).tolist(),
                "subspace_sin_median": [float(np.median(subspace_sin(res[a]["V_u"], res[b]["V_u"], m)))
                                        for m in range(1, 9)],
            }
    return out


def plots(res, summ, args):
    codes = list(res)
    os.makedirs(args.out, exist_ok=True)
    k = np.arange(1, 10)

    # 1. eigenvalue ladder (log coordinates)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, c in enumerate(codes):
        L = np.log10(np.maximum(res[c]["lam_log"], 1e-300))
        x = k + (i - 1) * 0.22
        q = np.percentile(L, [2.5, 16, 50, 84, 97.5], 0)
        ax.vlines(x, q[0], q[4], color=CODE_COLOR[c], lw=0.8, alpha=0.6)
        ax.vlines(x, q[1], q[3], color=CODE_COLOR[c], lw=4, alpha=0.6)
        ax.plot(x, q[2], "_", color="k", ms=8)
        ax.plot([], [], color=CODE_COLOR[c], lw=4, label=CODE_LABEL[c])
    for s, ls in zip((30, 50), ("--", ":")):
        ax.axhline(np.log10(s**2), color="k", lw=0.8, ls=ls, label=f"λ = σ², σ = {s} ppm (1σ width = 1 dex)")
        ax.axhline(np.log10(100 * s**2), color="grey", lw=0.8, ls=ls)
    ax.set_xticks(k, [f"v{j}" for j in k])
    ax.set_xlabel("Fisher eigendirection (stiffest → sloppiest)")
    ax.set_ylabel("log₁₀ λ  [ppm² per dex²]")
    ax.set_title("Eigenvalues of JᵀJ in log coordinates (bars: 68% and 95% of planets)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.text(9.4, np.log10(100 * 30**2), "0.1 dex", fontsize=7, color="grey", va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "eigenvalues.png"), dpi=150)
    plt.close(fig)

    # 2. eigenvector composition
    fig, axs = plt.subplots(1, len(codes), figsize=(4.2 * len(codes), 4), squeeze=False)
    for ax, c in zip(axs[0], codes):
        comp = np.median(res[c]["V_log"] ** 2, 0).T
        im = ax.imshow(comp, cmap="Greys", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(comp):
            if v >= 0.1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="w" if v > 0.5 else "k")
        ax.set_xticks(range(9), PARAM_NAMES, rotation=60, fontsize=8)
        ax.set_yticks(range(9), [f"v{j}" for j in k], fontsize=8)
        ax.set_title(f"{CODE_LABEL[c]}: median squared component", fontsize=9)
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
    fig.savefig(os.path.join(args.out, "eigenvectors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. N_eff vs sigma and posterior widths
    sig = np.geomspace(3, 300, 40)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for c in codes:
        lu = res[c]["lam_u"]
        ne = (lu[None] / (lu[None] + PRIOR_PRECISION * sig[:, None, None] ** 2)).sum(-1)
        q = np.percentile(ne, [16, 50, 84], 1)
        axs[0].fill_between(sig, q[0], q[2], color=CODE_COLOR[c], alpha=0.2)
        axs[0].plot(sig, q[1], color=CODE_COLOR[c], label=CODE_LABEL[c])
    axs[0].set_xscale("log")
    axs[0].set_xlabel("noise σ per bin (ppm)")
    axs[0].set_ylabel("effective number of constrained directions")
    axs[0].set_ylim(0, 9)
    axs[0].legend(fontsize=8)
    w = 0.8 / len(codes)
    for i, c in enumerate(codes):
        v = summ[c]["posterior_width_over_prior_width"][f"{args.sigmas[1]:g}"]
        axs[1].bar(np.arange(9) + (i - 1) * w, [v[p] for p in PARAM_NAMES], w, color=CODE_COLOR[c],
                   label=CODE_LABEL[c])
    axs[1].axhline(1, color="k", lw=0.8)
    axs[1].set_xticks(range(9), PARAM_NAMES)
    axs[1].set_ylabel("posterior width / prior width (median)")
    axs[1].set_title(f"Gaussian (Fisher + uniform prior) marginal widths, σ = {args.sigmas[1]:g} ppm", fontsize=9)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "constrained_directions.png"), dpi=150)
    plt.close(fig)

    # 4. eigenvalues vs feature height (POS shown; others in summary.json)
    fig, axs = plt.subplots(1, len(codes), figsize=(4.2 * len(codes), 3.8), squeeze=False, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, c in zip(axs[0], codes):
        L = np.log10(np.maximum(res[c]["lam_log"], 1e-300))
        f = res[c]["feat_ppm"]
        bins = np.geomspace(max(f.min(), 1), f.max(), 16)
        mid = np.sqrt(bins[1:] * bins[:-1])
        kk = np.digitize(f, bins) - 1
        for j in range(9):
            med = [np.median(L[kk == b, j]) if (kk == b).sum() >= 10 else np.nan for b in range(len(mid))]
            ax.plot(mid, med, color=cmap(j / 8), label=f"v{j + 1}")
        ax.axvline(210, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("feature height max − min (ppm)")
        ax.set_title(CODE_LABEL[c], fontsize=9)
    axs[0, 0].set_ylabel("median log₁₀ λ_k  [ppm² per dex²]")
    axs[0, -1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "eigenvalues_vs_features.png"), dpi=150)
    plt.close(fig)

    # 5. hyper-ribbon widths
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, c in enumerate(codes):
        W = np.log10(res[c]["width_arc"])
        x = k + (i - 1) * 0.22
        q = np.percentile(W, [16, 50, 84], 0)
        ax.vlines(x, q[0], q[2], color=CODE_COLOR[c], lw=4, alpha=0.6, label=CODE_LABEL[c])
        ax.plot(x, q[1], "_", color="k", ms=8)
    for s, ls in zip((30, 50), ("--", ":")):
        ax.axhline(np.log10(s), color="k", lw=0.8, ls=ls, label=f"σ = {s} ppm")
    ax.set_xticks(k, [f"v{j}" for j in k])
    ax.set_xlabel("unit-box Fisher eigendirection (stiffest → sloppiest)")
    ax.set_ylabel("log₁₀ width in data space (ppm)")
    ax.set_title("Width of the surface along each eigendirection, box face to box face (16–84%)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "widths.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", nargs="+", default=CODES)
    ap.add_argument("--n-points", type=int, default=4000)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[10, 30, 50, 100])
    ap.add_argument("--feature-bins", nargs="+", type=float, default=[30, 100, 210, 500, 1000, 3000])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emulator-dir", default="runs/emulator")
    ap.add_argument("--out", default="runs/fisher")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    lo, hi = box(args.data_dir)
    idx = eval_points(args.n_points, args.data_dir)
    res, summ = {}, {}
    for c in args.codes:
        res[c] = analyse(c, idx, lo, hi, args)
        summ[c] = summarise(res[c], args)
        np.savez_compressed(os.path.join(args.out, f"{c}_fisher.npz"), index=idx,
                            **{k: v for k, v in res[c].items() if k not in ("V_u_members",)})
    summ["code_comparison"] = compare_codes(res)
    summ["settings"] = vars(args)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    plots(res, summ, args)


if __name__ == "__main__":
    main()
