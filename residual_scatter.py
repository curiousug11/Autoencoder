"""Scatter plots of the per-planet difference between two codes against each planet parameter.

Every planet has a noiseless spectrum from each code, so for a pair (a, b) the difference
spectrum a - b (ppm) of the same planet isolates the code systematics. Each difference is
summarised by one number per planet:
  rms:      RMS of the difference after removing its wavelength mean (shape residual, ppm),
  offset:   wavelength mean of the difference (ppm); degenerate with Rp in a fit,
  relative: rms divided by the pair's mean feature height (%),
  zmse:     MSE between the two z-normalised spectra (each scaled by its own mean and std),
            i.e. the shape difference the autoencoders and the random forest see.
Each figure has one row per code pair and one column per parameter (plus log g), with the
binned median and 16-84% / 90th-percentile curves on top of the scatter. Further figures map
the median rms and zmse over T and log g.

Planets: those passing the 210 ppm feature-height cut in all codes (as in anomaly.py), or
all planets with --no-cut.

Usage:
    python residual_scatter.py
    python residual_scatter.py --no-cut --out runs/residuals/scatter_nocut
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from anomaly import LABELS
from data import DATA_DIR, DATASETS, feature_height, load_raw, znormalise
from poor_recon import spearman

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARAMS = ["Rs", "Rp", "Mp", "T", "H2O", "CH4", "CO", "CO2", "NH3"]
AXIS_LABELS = {"Rs": "Rs (R☉)", "Rp": "Rp (R_J)", "Mp": "Mp (M_J)", "T": "T (K)",
               "H2O": "log X(H₂O)", "CH4": "log X(CH₄)", "CO": "log X(CO)", "CO2": "log X(CO₂)",
               "NH3": "log X(NH₃)", "log_g": "log g (Mp/Rp², Jupiter units)"}
METRICS = {"rms": "shape residual RMS (ppm)", "offset": "mean offset (ppm)",
           "relative": "RMS / feature height (%)", "zmse": "z-normalised MSE"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/residuals/scatter")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--min-feature-height-ppm", type=float, default=210.0)
    p.add_argument("--no-cut", action="store_true", help="use all planets, not just those passing the cut")
    p.add_argument("--n-bins", type=int, default=20, help="quantile bins for the median curves")
    p.add_argument("--top-frac", type=float, default=0.05,
                   help="fraction of planets with the largest rms summarised in the printed table")
    return p.parse_args()


def pair_metrics(raw_a, raw_b):
    """Per-planet rms, offset, relative residual and z-normalised MSE for spectra in ppm, (n, bins) each."""
    diff = raw_a - raw_b
    offset = diff.mean(1)
    rms = np.sqrt(((diff - offset[:, None]) ** 2).mean(1))
    fh = (feature_height(raw_a) + feature_height(raw_b)) / 2
    zmse = ((znormalise(raw_a)[0] - znormalise(raw_b)[0]) ** 2).mean(1)
    return {"rms": rms, "offset": offset, "relative": 100 * rms / fh, "zmse": zmse}


def binned_percentiles(x, y, n_bins, qs=(16, 50, 84, 90)):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centres = np.array([np.median(x[which == i]) for i in range(len(edges) - 1)])
    pct = np.array([np.percentile(y[which == i], qs) for i in range(len(edges) - 1)])
    return centres, {q: pct[:, k] for k, q in enumerate(qs)}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw, params = {}, None
    for d in args.datasets:
        raw[d], params, _ = load_raw(args.data_dir, d)
    keep = np.ones(len(params), bool)
    if not args.no_cut:
        for d in args.datasets:
            keep &= feature_height(raw[d]) >= args.min_feature_height_ppm * 1e-6
    raw = {d: s[keep] * 1e6 for d, s in raw.items()}  # ppm
    params = params[keep]
    x = {name: params[:, j] for j, name in enumerate(PARAMS)}
    x["log_g"] = np.log10(x["Mp"] / x["Rp"] ** 2)
    columns = PARAMS + ["log_g"]
    pairs = [(a, b) for i, a in enumerate(args.datasets) for b in args.datasets[i + 1:]]
    metrics = {(a, b): pair_metrics(raw[a], raw[b]) for a, b in pairs}
    print(f"{keep.sum()} planets")

    results = {"n_planets": int(keep.sum()), "top_frac": args.top_frac, "pairs": {}}
    n_top = int(round(args.top_frac * keep.sum()))
    for a, b in pairs:
        m = metrics[(a, b)]
        top = np.argsort(m["rms"])[::-1][:n_top]
        rest = np.setdiff1d(np.arange(len(m["rms"])), top)
        res = {
            "rms_ppm": {"median": float(np.median(m["rms"])), "p90": float(np.percentile(m["rms"], 90)),
                        "p99": float(np.percentile(m["rms"], 99))},
            "zmse": {"median": float(np.median(m["zmse"])), "p90": float(np.percentile(m["zmse"], 90)),
                     "p99": float(np.percentile(m["zmse"], 99))},
            "offset_ppm": {"median": float(np.median(m["offset"])), "p16": float(np.percentile(m["offset"], 16)),
                           "p84": float(np.percentile(m["offset"], 84))},
            "spearman": {k: {c: spearman(x[c], m[k]) for c in columns} for k in METRICS},
            "top_vs_rest_median": {c: [float(np.median(x[c][top])), float(np.median(x[c][rest]))] for c in columns},
        }
        results["pairs"][f"{a}-{b}"] = res
        report(a, b, res, columns, n_top)
    (out / "residual_scatter.json").write_text(json.dumps(results, indent=2))

    for k, ylabel in METRICS.items():
        plot_scatter(args, pairs, metrics, x, columns, k, ylabel, out / f"residual_scatter_{k}.png")
    plot_map(args, pairs, metrics, x, "rms", out / "residual_map_T_logg.png")
    plot_map(args, pairs, metrics, x, "zmse", out / "residual_map_zmse_T_logg.png")
    print(f"\nsaved to {out}/")


def report(a, b, res, columns, n_top):
    r, o = res["rms_ppm"], res["offset_ppm"]
    print(f"\n=== {LABELS[a]} - {LABELS[b]} ===")
    print(f"rms: median {r['median']:.1f} ppm, 90th pct {r['p90']:.1f}, 99th pct {r['p99']:.1f}; "
          f"offset: median {o['median']:+.1f} ppm (16-84%: {o['p16']:+.1f} to {o['p84']:+.1f})")
    print(f"  {'':>8} {'rho(rms)':>9} {'rho(offset)':>12} {'rho(rel)':>9} {'rho(zmse)':>10}   "
          f"median: top {n_top} by rms vs rest")
    for c in columns:
        s = res["spearman"]
        t, rest = res["top_vs_rest_median"][c]
        print(f"  {c:>8} {s['rms'][c]:>+9.2f} {s['offset'][c]:>+12.2f} {s['relative'][c]:>+9.2f} "
              f"{s['zmse'][c]:>+10.2f}   "
              f"{t:>9.3g} vs {rest:<9.3g}")


def plot_scatter(args, pairs, metrics, x, columns, key, ylabel, path):
    fig, axes = plt.subplots(len(pairs), len(columns), figsize=(2.3 * len(columns), 2.5 * len(pairs)),
                             squeeze=False, sharey="row")
    for i, (a, b) in enumerate(pairs):
        y = metrics[(a, b)][key]
        for j, c in enumerate(columns):
            ax = axes[i, j]
            ax.scatter(x[c], y, s=1, color="grey", alpha=0.15, lw=0, rasterized=True)
            centres, pct = binned_percentiles(x[c], y, args.n_bins)
            ax.fill_between(centres, pct[16], pct[84], color="C0", alpha=0.3, lw=0)
            ax.plot(centres, pct[50], color="k", lw=1.3)
            ax.plot(centres, pct[90], color="C3", lw=1, ls="--")
            if key == "offset":
                ax.set_yscale("symlog", linthresh=10)
                ax.axhline(0, color="k", lw=0.6)
            else:
                ax.set_yscale("log")
            rho = spearman(x[c], y)
            ax.text(0.03, 0.95, f"ρ={rho:+.2f}", transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(facecolor="w", alpha=0.7, lw=0))
            ax.tick_params(labelsize=6)
            if i == len(pairs) - 1:
                ax.set_xlabel(AXIS_LABELS[c], fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{LABELS[a]} − {LABELS[b]}\n{ylabel}", fontsize=7)
    handles = [plt.Line2D([], [], color="k", lw=1.3, label="median"),
               plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.3, label="16–84%"),
               plt.Line2D([], [], color="C3", lw=1, ls="--", label="90th percentile")]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)
    fig.suptitle(f"Per-planet {ylabel} between codes vs planet parameters "
                 f"({len(metrics[pairs[0]][key])} planets; ρ = Spearman)", fontsize=10, x=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_map(args, pairs, metrics, x, key, path):
    """Median of metric `key` in bins of T and log g, one panel per pair."""
    t, g = x["T"], x["log_g"]
    tb = np.linspace(t.min(), t.max(), 16)
    gb = np.linspace(np.percentile(g, 0.5), np.percentile(g, 99.5), 16)
    ti = np.clip(np.searchsorted(tb, t, side="right") - 1, 0, len(tb) - 2)
    gi = np.clip(np.searchsorted(gb, g, side="right") - 1, 0, len(gb) - 2)
    inside = (g >= gb[0]) & (g <= gb[-1])
    maps = []
    for a, b in pairs:
        y = metrics[(a, b)][key]
        med = np.full((len(tb) - 1, len(gb) - 1), np.nan)
        for p in range(len(tb) - 1):
            for q in range(len(gb) - 1):
                sel = inside & (ti == p) & (gi == q)
                if sel.sum() >= 20:
                    med[p, q] = np.median(y[sel])
        maps.append(med)
    vmin, vmax = np.nanmin(maps), np.nanmax(maps)
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2), squeeze=False)
    for ax, (a, b), med in zip(axes[0], pairs, maps):
        im = ax.pcolormesh(tb, gb, med.T, cmap="magma_r", norm=matplotlib.colors.LogNorm(vmin, vmax))
        ax.set(xlabel="T (K)", title=f"{LABELS[a]} − {LABELS[b]}")
    axes[0, 0].set_ylabel(AXIS_LABELS["log_g"])
    fig.colorbar(im, ax=axes[0].tolist(), label=f"median {METRICS[key]}")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
