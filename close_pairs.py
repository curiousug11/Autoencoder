"""Step 1 of the spectral-manifold project: a model-free scan for global non-uniqueness.

For each code (POS, petitRADTRANS, TauREx) we look for pairs of spectra that are
statistically indistinguishable at white noise sigma, i.e.

    ||S_i - S_j||^2 / sigma^2 < chi2_95(9) = 16.92,

so that, if S_i were observed with noise sigma, theta_j would sit inside the 95%
region of a 9-parameter posterior. Only the 98,725 sampled spectra are used:
no emulator and no new radiative transfer.

Outputs:
  * how often a spectrum has such a partner, split by feature height
    (featureless spectra collapse onto the flat-line boundary of the manifold);
  * which parameters stay pinned and which slide within matched pairs;
  * the sampling resolution: the misfit of the cheapest one-step move to a
    nearest parameter-space neighbour. This shows whether the samples alone can
    tell one connected valley from two separate basins (they cannot: one step
    costs far more than the threshold);
  * the list of feature-rich matched pairs, which step 2 (an emulator) needs to
    test for a barrier along the path between them.

    python close_pairs.py                      # all codes, sigma = 10 20 30 50 ppm
    python close_pairs.py --codes pos --max-refs 500
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

PARAM_NAMES = ["Rs", "Rp", "Mp", "T", "chem1", "chem2", "chem3", "chem4", "chem5"]
CODE_LABEL = {"pos": "POS", "prt": "petitRADTRANS", "taurex": "TauREx"}
CHI2_95 = float(chi2.ppf(0.95, df=9))  # 16.92


def load(code, data_dir):
    S = np.load(os.path.join(data_dir, f"{code}_spec_binned.npy")).astype(np.float64)
    P = np.load(os.path.join(data_dir, f"{code}_params_clean.npy")).astype(np.float64)
    return S, P


def unit_params(P):
    lo, hi = P.min(0), P.max(0)
    return (P - lo) / (hi - lo)


def derived(P):
    Rs, Rp, Mp, T = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return {
        "log Rp/Rs": np.log10(Rp / Rs),
        # feature amplitude ~ 2 Rp H / Rs^2 with H ~ T Rp^2 / Mp
        "log T Rp^3/(Mp Rs^2)": np.log10(T * Rp**3 / (Mp * Rs**2)),
        "mean log abundance": P[:, 4:9].mean(1),
    }


def find_pairs(S, thr, chunk=1000):
    """All pairs i<j with ||S_i - S_j||^2 < thr."""
    n = len(S)
    sq = (S**2).sum(1)
    I, J, D = [], [], []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d2 = sq[a:b, None] + sq[None, :] - 2.0 * S[a:b] @ S.T
        ii, jj = np.nonzero(d2 < thr)
        keep = (ii + a) < jj
        ii, jj = ii[keep], jj[keep]
        I.append((ii + a).astype(np.int32))
        J.append(jj.astype(np.int32))
        D.append(d2[ii, jj])
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def knn_index(U, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(U)
    _, idx = nn.kneighbors(U)
    return idx[:, 1:]


def one_step_misfit(S, nbr, chunk=5000):
    """Misfit ||S_nbr - S_i||^2 of the cheapest move to one of i's parameter-space neighbours."""
    out = np.empty(len(S))
    for a in range(0, len(S), chunk):
        b = min(len(S), a + chunk)
        out[a:b] = ((S[nbr[a:b]] - S[a:b, None, :]) ** 2).sum(-1).min(1)
    return out


def analyse_code(code, args, U, nbr, rng):
    t0 = time.time()
    S, P = load(code, args.data_dir)
    n = len(S)
    feat = (S.max(1) - S.min(1)) * 1e6
    rich = feat >= args.feature_cut
    sig = np.array(sorted(args.sigmas), dtype=float)
    thr = CHI2_95 * (sig * 1e-6) ** 2

    I, J, D = find_pairs(S, thr.max())
    print(f"[{code}] {len(I):,} pairs within {sig.max():.0f} ppm ({time.time() - t0:.0f}s)")

    # random feature-rich pairs as a baseline for parameter spreads
    ridx = np.flatnonzero(rich)
    ra, rb = rng.choice(ridx, 200000), rng.choice(ridx, 200000)
    der = derived(P)

    summary = {"code": code, "n_spectra": n, "n_feature_rich": int(rich.sum()),
               "feature_cut_ppm": args.feature_cut, "chi2_95_9dof": CHI2_95, "per_sigma": {}}
    for s, t in zip(sig, thr):
        m = D < t
        deg = np.bincount(np.r_[I[m], J[m]], minlength=n)
        mr = m & rich[I] & rich[J]
        entry = {
            "threshold_rms_ppm_per_bin": float(np.sqrt(t / S.shape[1]) * 1e6),
            "n_pairs_all": int(m.sum()),
            "frac_with_partner_all": float(np.mean(deg > 0)),
            "frac_with_partner_featureless": float(np.mean(deg[~rich] > 0)),
            "frac_with_partner_feature_rich": float(np.mean(deg[rich] > 0)),
            "n_pairs_feature_rich": int(mr.sum()),
        }
        if mr.any():
            a, b = I[mr], J[mr]
            entry["param_dist_pct_16_50_84"] = np.percentile(
                np.linalg.norm(U[a] - U[b], axis=1), [16, 50, 84]).round(3).tolist()
            entry["random_param_dist_median"] = float(np.median(np.linalg.norm(U[ra] - U[rb], axis=1)))
            entry["median_abs_dU_pair"] = {p: float(np.median(np.abs(U[a, k] - U[b, k])))
                                           for k, p in enumerate(PARAM_NAMES)}
            entry["median_abs_dU_random"] = {p: float(np.median(np.abs(U[ra, k] - U[rb, k])))
                                             for k, p in enumerate(PARAM_NAMES)}
            entry["median_abs_derived_pair"] = {k: float(np.median(np.abs(v[a] - v[b]))) for k, v in der.items()}
            entry["median_abs_derived_random"] = {k: float(np.median(np.abs(v[ra] - v[rb]))) for k, v in der.items()}
            # species swaps: one abundance up by >= 2 dex while another drops by >= 2 dex
            dc = P[b, 4:9] - P[a, 4:9]
            swap = (dc.max(1) >= 2) & (dc.min(1) <= -2)
            counts = {}
            for u, d in zip(dc[swap].argmax(1), dc[swap].argmin(1)):
                key = "-".join(PARAM_NAMES[4 + x] for x in sorted((u, d)))
                counts[key] = counts.get(key, 0) + 1
            entry["frac_species_swap_2dex"] = float(swap.mean())
            entry["species_swap_counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        summary["per_sigma"][f"{s:g}"] = entry

    # ---- sampling resolution ---------------------------------------------------------------
    step = one_step_misfit(S, nbr)
    summary["sampling"] = {"graph_k": args.k, "per_sigma": {}}
    for s in sig:
        x = step[rich] / (s * 1e-6) ** 2
        summary["sampling"]["per_sigma"][f"{s:g}"] = {
            "median_one_step_dchi2_feature_rich": float(np.median(x)),
            "frac_one_step_below_threshold_feature_rich": float(np.mean(x < CHI2_95)),
        }

    # ---- feature-rich matched pairs for step 2 ------------------------------------------------
    mr = rich[I] & rich[J]
    I, J, D = I[mr], J[mr], D[mr]
    sigma_min = np.sqrt(D / CHI2_95) * 1e6  # smallest noise at which the pair is indistinguishable
    dist = np.linalg.norm(U[I] - U[J], axis=1)
    o = np.argsort(sigma_min)
    I, J, sigma_min, dist = I[o], J[o], sigma_min[o], dist[o]
    np.savez(os.path.join(args.out, f"{code}_pairs.npz"), i=I, j=J, sigma_min_ppm=sigma_min, param_dist=dist)
    with open(os.path.join(args.out, f"{code}_pairs_top.csv"), "w") as f:
        f.write("sigma_min_ppm,param_dist,i,j," + ",".join(f"i_{p}" for p in PARAM_NAMES) + ","
                + ",".join(f"j_{p}" for p in PARAM_NAMES) + "\n")
        for q in range(min(args.n_csv, len(I))):
            f.write(f"{sigma_min[q]:.2f},{dist[q]:.3f},{I[q]},{J[q]},"
                    + ",".join(f"{x:.5g}" for x in P[I[q]]) + "," + ",".join(f"{x:.5g}" for x in P[J[q]]) + "\n")
    summary["n_feature_rich_pairs_exported"] = int(len(I))
    print(f"[{code}] done in {time.time() - t0:.0f}s; {len(I)} feature-rich pairs")
    return summary, S, P, (I, J, sigma_min, dist)


def plot_all(results, args):
    sig = sorted(args.sigmas)
    codes = list(results)
    colors = {"pos": "#1f77b4", "prt": "#d62728", "taurex": "#2ca02c"}

    # 1. how often a spectrum has an indistinguishable partner
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in codes:
        ps = results[c][0]["per_sigma"]
        ax.plot(sig, [ps[f"{s:g}"]["frac_with_partner_featureless"] for s in sig], "--o",
                color=colors[c], label=f"{CODE_LABEL[c]}: feature < {args.feature_cut:g} ppm")
        ax.plot(sig, [ps[f"{s:g}"]["frac_with_partner_feature_rich"] for s in sig], "-o",
                color=colors[c], label=f"{CODE_LABEL[c]}: feature ≥ {args.feature_cut:g} ppm")
    ax.set_xlabel("noise σ per bin (ppm)")
    ax.set_ylabel("fraction with an indistinguishable partner")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Spectra matched by another sample at Δχ² < 16.9")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "partner_fraction.png"), dpi=150)
    plt.close(fig)

    # 2. which parameters stay pinned in matched pairs
    s_ref = f"{args.spread_sigma:g}"
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [9, 3]})
    w = 0.8 / len(codes)
    for k, c in enumerate(codes):
        e = results[c][0]["per_sigma"][s_ref]
        if "median_abs_dU_pair" not in e:
            continue
        ratio = [e["median_abs_dU_pair"][p] / e["median_abs_dU_random"][p] for p in PARAM_NAMES]
        axs[0].bar(np.arange(9) + (k - (len(codes) - 1) / 2) * w, ratio, w, color=colors[c], label=CODE_LABEL[c])
        dk = list(e["median_abs_derived_pair"])
        dr = [e["median_abs_derived_pair"][q] / e["median_abs_derived_random"][q] for q in dk]
        axs[1].bar(np.arange(len(dk)) + (k - (len(codes) - 1) / 2) * w, dr, w, color=colors[c])
    axs[0].set_xticks(range(9), PARAM_NAMES)
    axs[0].axhline(1, color="k", lw=0.8)
    axs[0].set_ylabel("median |Δ| in matched pairs / random pairs")
    axs[0].set_title(f"Parameter spread within matched pairs (σ = {s_ref} ppm)")
    axs[0].legend(fontsize=8)
    axs[1].set_xticks(range(len(dk)), ["Rp/Rs", "T Rp³/(Mp Rs²)", "mean log X"], rotation=20)
    axs[1].axhline(1, color="k", lw=0.8)
    axs[1].set_yscale("log")
    axs[1].set_title("Derived combinations")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "param_spread.png"), dpi=150)
    plt.close(fig)

    # 3. sampling resolution: can the samples resolve a valley?
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in codes:
        sp = results[c][0]["sampling"]["per_sigma"]
        ax.plot(sig, [sp[f"{s:g}"]["median_one_step_dchi2_feature_rich"] for s in sig], "-o",
                color=colors[c], label=CODE_LABEL[c])
    ax.axhline(CHI2_95, color="k", lw=0.8, ls="--", label="Δχ² threshold (16.9)")
    ax.set_yscale("log")
    ax.set_xlabel("noise σ per bin (ppm)")
    ax.set_ylabel("median Δχ² of one step to a neighbour")
    ax.set_title(f"Sampling resolution (feature-rich spectra, k = {args.k})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sampling_resolution.png"), dpi=150)
    plt.close(fig)

    # 4. examples: the most distant pair that is indistinguishable at spread_sigma
    fig, axs = plt.subplots(2, len(codes), figsize=(4.4 * len(codes), 5.5), squeeze=False,
                            gridspec_kw={"height_ratios": [3, 1.3]})
    wl = np.load(os.path.join(args.data_dir, "wavelength_binned.npy"))
    s = args.spread_sigma
    for k, c in enumerate(codes):
        _, S, P, (I, J, smin, dist) = results[c]
        m = smin < s
        if not m.any():
            axs[0, k].set_title(f"{CODE_LABEL[c]}: no pairs at {s:g} ppm")
            continue
        q = np.flatnonzero(m)[np.argmax(dist[m])]
        i, j = I[q], J[q]
        axs[0, k].plot(wl, S[i] * 1e6, "-", color="k", lw=1.2, label=f"#{i}")
        axs[0, k].plot(wl, S[j] * 1e6, "--", color=colors[c], lw=1.2, label=f"#{j}")
        axs[0, k].set_xscale("log")
        axs[0, k].set_ylabel("transit depth (ppm)")
        axs[0, k].set_title(f"{CODE_LABEL[c]}: most distant pair at σ={s:g} ppm\n"
                            f"param distance {dist[q]:.2f}, indistinguishable down to {smin[q]:.0f} ppm",
                            fontsize=8)
        txt = "\n".join(f"{p:6s} {P[i, n]:8.3g} {P[j, n]:8.3g}" for n, p in enumerate(PARAM_NAMES))
        axs[0, k].text(0.02, 0.97, txt, transform=axs[0, k].transAxes, va="top", family="monospace", fontsize=6.5)
        axs[0, k].legend(fontsize=7, loc="lower right")
        axs[1, k].plot(wl, (S[j] - S[i]) * 1e6, color=colors[c])
        axs[1, k].axhspan(-s, s, color="0.85")
        axs[1, k].set_xscale("log")
        axs[1, k].set_xlabel("wavelength (µm)")
        axs[1, k].set_ylabel("Δ (ppm)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "distant_pair_examples.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", nargs="+", default=["pos", "prt", "taurex"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[10, 20, 30, 50], help="noise per bin, ppm")
    ap.add_argument("--feature-cut", type=float, default=210.0, help="max-min of spectrum, ppm")
    ap.add_argument("--k", type=int, default=15, help="parameter-space neighbours for the sampling-resolution check")
    ap.add_argument("--n-csv", type=int, default=500, help="most indistinguishable pairs written to CSV, per code")
    ap.add_argument("--spread-sigma", type=float, default=30.0, help="sigma used for the spread and example plots")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/close_pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    P0 = load(args.codes[0], args.data_dir)[1]
    for c in args.codes[1:]:
        assert np.array_equal(P0, load(c, args.data_dir)[1]), "parameter files differ between codes"
    U = unit_params(P0)
    t = time.time()
    nbr = knn_index(U, args.k)
    print(f"kNN graph k={args.k} built ({time.time() - t:.0f}s)")

    results = {}
    for c in args.codes:
        results[c] = analyse_code(c, args, U, nbr, rng)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({c: r[0] for c, r in results.items()} | {"args": vars(args)}, f, indent=2)
    plot_all(results, args)


if __name__ == "__main__":
    main()
