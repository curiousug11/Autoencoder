"""Paired residual analysis: where and why do the codes' spectra differ?

Every planet has a noiseless spectrum from each code, so the code systematics can be measured
directly as difference spectra, e.g. POS - TauREx for the same planet.

For each pair of codes:
  1. Offset: the mean of the difference over wavelength (ppm). It is degenerate with the planet
     radius in a fit, so it is reported separately and removed before steps 2-4.
  2. Residual spectrum vs wavelength: median and percentile bands of the mean-subtracted
     difference, in ppm ("absolute") and divided by the pair's mean feature height ("relative").
  3. PCA of the residuals: a few components usually capture most of the code-to-code
     difference; their shapes say which spectral regions disagree.
  4. What drives each component: Spearman correlation of the component scores with the planet
     parameters (+ log g, scale-height proxy, feature height), and the out-of-sample R^2 of a
     random-forest regression of the score on the 9 parameters.

Planets: those passing the 210 ppm feature-height cut in all codes (as in anomaly.py), or
all planets with --no-cut.

Usage:
    python residual_analysis.py
    python residual_analysis.py --n-components 6 --no-cut
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from anomaly import LABELS
from data import DATA_DIR, DATASETS, feature_height, load_raw
from poor_recon import DERIVED, features, spearman

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARAM_NAMES = ["Rs", "Rp", "Mp", "T", "chem1", "chem2", "chem3", "chem4", "chem5"]
# Approximate centres of strong bands, for orientation only.
BANDS = {"H2O": [1.15, 1.4, 1.9, 2.7, 6.3], "CH4": [1.7, 2.3, 3.3], "CO2": [2.0, 4.3],
         "CO": [2.35, 4.6], "NH3": [1.5, 2.0, 3.0]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/residuals")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--min-feature-height-ppm", type=float, default=210.0)
    p.add_argument("--no-cut", action="store_true", help="use all planets, not just those passing the cut")
    p.add_argument("--n-components", type=int, default=4, help="components to describe and regress")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pca(x):
    """PCA via SVD on centred data. Components are signed so the largest loading is positive."""
    mean = x.mean(0)
    u, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])
    vt *= signs[:, None]
    scores = (x - mean) @ vt.T
    var = s**2 / (len(x) - 1)
    return mean, vt, scores, var / var.sum()


def regress_r2(params, y, seed):
    """Out-of-sample R^2 (30% held out) of a random forest predicting y from the parameters."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_test = int(0.3 * len(y))
    te, tr = perm[:n_test], perm[n_test:]
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.5, n_jobs=4, random_state=seed)
    rf.fit(params[tr], y[tr])
    pred = rf.predict(params[te])
    r2 = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return float(r2), rf.feature_importances_


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw, params, wavelength = {}, None, None
    for d in args.datasets:
        raw[d], params, wavelength = load_raw(args.data_dir, d)
    keep = np.ones(len(params), bool)
    if not args.no_cut:
        for d in args.datasets:
            keep &= feature_height(raw[d]) >= args.min_feature_height_ppm * 1e-6
    raw = {d: s[keep] * 1e6 for d, s in raw.items()}  # ppm
    params = params[keep]
    fh = {d: feature_height(raw[d]) for d in args.datasets}
    order = np.argsort(wavelength)
    wl = wavelength[order]
    feat = features(params, np.mean(list(fh.values()), 0) * 1e-6 * 1e6)  # mean feature height, ppm
    feat_names = PARAM_NAMES + DERIVED
    pairs = [(a, b) for i, a in enumerate(args.datasets) for b in args.datasets[i + 1:]]
    print(f"{keep.sum()} planets")

    results = {"n_planets": int(keep.sum()), "pairs": {}}
    pair_data = {}
    for a, b in pairs:
        key = f"{a}-{b}"
        diff = raw[a] - raw[b]
        offset = diff.mean(1)
        resid = diff - offset[:, None]
        rel = resid / ((fh[a] + fh[b]) / 2)[:, None]
        rms = np.sqrt((resid**2).mean(1))
        res = {
            "offset_ppm": {"median": float(np.median(offset)), "p16": float(np.percentile(offset, 16)),
                           "p84": float(np.percentile(offset, 84)),
                           "spearman": {n: spearman(feat[:, j], offset) for j, n in enumerate(feat_names)}},
            "residual_rms_ppm": {"median": float(np.median(rms)), "p90": float(np.percentile(rms, 90))},
            "relative_rms_median": float(np.median(np.sqrt((rel**2).mean(1)))),
        }
        variants = {}
        for vname, x in (("absolute", resid), ("relative", rel)):
            mean, comps, scores, ratio = pca(x)
            cum = np.cumsum(ratio)
            comp_res = []
            for k in range(args.n_components):
                r2, imp = regress_r2(params, scores[:, k], args.seed)
                comp_res.append({
                    "explained_variance": float(ratio[k]),
                    "loading": comps[k][order].tolist(),
                    "spearman": {n: spearman(feat[:, j], scores[:, k]) for j, n in enumerate(feat_names)},
                    "rf_r2_from_params": r2,
                    "rf_importance": dict(zip(PARAM_NAMES, imp.round(4).tolist())),
                })
            variants[vname] = {"mean_residual": mean[order].tolist(),
                               "n_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
                               "n_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
                               "cumulative": cum[:10].tolist(), "components": comp_res}
            pair_data[(key, vname)] = (x, scores, ratio, comps)
        res["pca"] = variants
        results["pairs"][key] = res
        report(key, res, args.n_components)

    results["wavelength_um"] = wl.tolist()
    (out / "residual_results.json").write_text(json.dumps(results, indent=2))
    plot_residual_bands(pairs, raw, wl, order, out / "residual_spectra.png")
    for vname in ("absolute", "relative"):
        plot_components(args, pairs, pair_data, results, wl, order, vname, out / f"pca_components_{vname}.png")
        plot_drivers(args, pairs, results, feat_names, vname, out / f"pca_drivers_{vname}.png")
    plot_offsets(pairs, raw, feat, feat_names, out / "offsets.png")
    print(f"\nsaved to {out}/")


def report(key, res, n_comp):
    o = res["offset_ppm"]
    top_o = max(o["spearman"], key=lambda n: abs(o["spearman"][n]))
    print(f"\n=== {key} ===")
    print(f"offset: median {o['median']:.1f} ppm (16-84%: {o['p16']:.1f} to {o['p84']:.1f}); "
          f"strongest driver {top_o} (rho={o['spearman'][top_o]:+.2f})")
    print(f"residual RMS after removing offset: median {res['residual_rms_ppm']['median']:.1f} ppm, "
          f"90th pct {res['residual_rms_ppm']['p90']:.1f} ppm; relative to feature height: "
          f"{res['relative_rms_median']:.2%}")
    for vname, v in res["pca"].items():
        print(f"  PCA ({vname}): {v['n_for_90pct']} comps for 90%, {v['n_for_99pct']} for 99%; "
              f"first 5 cumulative " + " ".join(f"{c:.2f}" for c in v["cumulative"][:5]))
        for k, c in enumerate(v["components"][:n_comp]):
            top = sorted(c["spearman"], key=lambda n: -abs(c["spearman"][n]))[:3]
            imp = sorted(c["rf_importance"], key=lambda n: -c["rf_importance"][n])[:3]
            print(f"    PC{k + 1}: {c['explained_variance']:.1%} var, R^2 from params {c['rf_r2_from_params']:.2f}; "
                  f"rho " + ", ".join(f"{n} {c['spearman'][n]:+.2f}" for n in top)
                  + f"; RF importance " + ", ".join(f"{n} {c['rf_importance'][n]:.2f}" for n in imp))


def mark_bands(ax):
    for i, (mol, centres) in enumerate(BANDS.items()):
        for c in centres:
            ax.axvline(c, color=f"C{i + 5}", lw=0.6, alpha=0.5)
        ax.plot([], [], color=f"C{i + 5}", lw=0.8, label=f"{mol} (approx.)")


def plot_residual_bands(pairs, raw, wl, order, path):
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3.3 * len(pairs)), squeeze=False)
    for i, (a, b) in enumerate(pairs):
        diff = raw[a] - raw[b]
        resid = diff - diff.mean(1, keepdims=True)
        fh = (np.ptp(raw[a], 1) + np.ptp(raw[b], 1)) / 2
        for ax, x, unit in ((axes[i, 0], resid, "ppm"), (axes[i, 1], resid / fh[:, None], "fraction of feature height")):
            x = x[:, order]
            p = np.percentile(x, [5, 16, 50, 84, 95], axis=0)
            ax.fill_between(wl, p[0], p[4], color="C0", alpha=0.15, label="5-95%")
            ax.fill_between(wl, p[1], p[3], color="C0", alpha=0.35, label="16-84%")
            ax.plot(wl, p[2], "k-", lw=1.2, label="median")
            ax.axhline(0, color="grey", lw=0.6)
            mark_bands(ax)
            ax.set(xscale="log", ylabel=unit, title=f"{LABELS[a]} − {LABELS[b]} (offset removed)")
            if i == 0 and unit == "ppm":
                ax.legend(fontsize=6, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (µm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_components(args, pairs, pair_data, results, wl, order, vname, path):
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3.3 * len(pairs)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    for i, (a, b) in enumerate(pairs):
        key = f"{a}-{b}"
        _, _, ratio, comps = pair_data[(key, vname)]
        ax = axes[i, 0]
        for k in range(args.n_components):
            ax.plot(wl, comps[k][order], "o-", ms=2.5, lw=1.2, color=f"C{k}",
                    label=f"PC{k + 1} ({ratio[k]:.0%})")
        ax.axhline(0, color="grey", lw=0.6)
        mark_bands(ax)
        ax.set(xscale="log", ylabel="loading", title=f"{LABELS[a]} − {LABELS[b]}: residual PCA ({vname})")
        ax.legend(fontsize=7, ncol=2)
        ax = axes[i, 1]
        cum = results["pairs"][key]["pca"][vname]["cumulative"]
        ax.plot(range(1, len(cum) + 1), cum, "ko-", ms=3)
        ax.axhline(0.9, color="grey", ls=":", lw=1)
        ax.axhline(0.99, color="grey", ls=":", lw=1)
        ax.set(ylim=(0, 1.01), xlabel="components", ylabel="cumulative variance")
    axes[-1, 0].set_xlabel("wavelength (µm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drivers(args, pairs, results, feat_names, vname, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.2), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        comps = results["pairs"][f"{a}-{b}"]["pca"][vname]["components"][:args.n_components]
        m = np.array([[c["spearman"][n] for n in feat_names] for c in comps])
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                if abs(m[r, c]) >= 0.2:
                    ax.text(c, r, f"{m[r, c]:+.1f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(feat_names)), feat_names, rotation=60, fontsize=7)
        ax.set_yticks(range(len(comps)),
                      [f"PC{k + 1} (R²={c['rf_r2_from_params']:.2f})" for k, c in enumerate(comps)], fontsize=7)
        ax.set_title(f"{LABELS[a]} − {LABELS[b]} ({vname})", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label="Spearman ρ (score vs quantity)", shrink=0.8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_offsets(pairs, raw, feat, feat_names, path):
    t, logg = feat[:, feat_names.index("T")], feat[:, feat_names.index("log_g")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        off = (raw[a] - raw[b]).mean(1)
        lim = np.percentile(np.abs(off), 98)
        sc = ax.scatter(t, logg, c=off, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
        fig.colorbar(sc, ax=ax, label="offset (ppm)")
        ax.set(xlabel="T", ylabel="log g", title=f"{LABELS[a]} − {LABELS[b]} mean offset")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
