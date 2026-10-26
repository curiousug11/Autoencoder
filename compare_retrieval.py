"""Compare two ml_retrieval.py runs, e.g. the same models tested on noiseless vs noisy spectra.

For every (training code, tested code, parameter) it shows R^2, the median bias and the 16-84%
half-width of predicted - true, the last two as % of the parameter's prior range.

If the runs differ in test noise, a second figure (vs_noise.png) shows each metric against the
test noise level, averaged over self-retrieval pairs and over cross-retrieval pairs.

Usage:
    python compare_retrieval.py runs/retrieval runs/retrieval_test10ppm --labels "0 ppm" "10 ppm"
    python compare_retrieval.py runs/retrieval runs/retrieval_test{10,20,30,50}ppm --out runs/retrieval_noise
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

from anomaly import LABELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT = {"pos": "POS", "prt": "pRT", "taurex": "TRx"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="ml_retrieval.py output directories")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default=None, help="output directory (default: the last run)")
    return p.parse_args()


def main():
    args = parse_args()
    res = [json.loads((Path(r) / "retrieval_results.json").read_text()) for r in args.runs]
    noise = [r.get("test_noise_ppm", r.get("noise_ppm", 0)) for r in res]
    labels = args.labels or [f"{n:g} ppm test" for n in noise]
    out_dir = Path(args.out) if args.out else Path(args.runs[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    params, rng = res[0]["params"], res[0]["prior_range"]
    combos = [(a, b) for a in res[0]["train_codes"] for b in res[0]["train_codes"][a][params[0]]["tested_on"]]

    def get(r, a, b, n, key):
        t = r["train_codes"][a][n]["tested_on"][b]
        if key == "r2":
            return t["r2"]
        return 100 * (t["median_bias"] if key == "bias" else t["spread_16_84"]) / rng[n]

    rows = [("r2", "R²"), ("bias", "median bias\n(% of range)"), ("spread", "16–84% half-width\n(% of range)")]
    fig, axes = plt.subplots(len(rows), len(params), figsize=(2.6 * len(params), 8.5), sharex=True)
    x = np.arange(len(combos))
    for j, n in enumerate(params):
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for k, (r, lab) in enumerate(zip(res, labels)):
                vals = [get(r, a, b, n, key) for a, b in combos]
                ax.scatter(x + (k - (len(res) - 1) / 2) * 0.25, vals, s=14, color=f"C{k}", label=lab, zorder=3)
            for xi, (a, b) in enumerate(combos):
                if a == b:
                    ax.axvspan(xi - 0.45, xi + 0.45, color="grey", alpha=0.12, lw=0)
            if key == "bias":
                ax.axhline(0, color="grey", lw=0.8)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(n)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xticks(x, [f"{SHORT.get(a, a)}→{SHORT.get(b, b)}" for a, b in combos], rotation=90, fontsize=6)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Retrieval: " + " vs ".join(labels) + " (grey = self retrieval; x = trained on → tested on)")
    fig.tight_layout()
    out = out_dir / "comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"{'trained→tested':<16}{'param':<7}" + "".join(f"{lab:>30}" for lab in labels))
    print(f"{'':<23}" + "".join(f"{'R²':>8}{'bias%':>10}{'spread%':>12}" for _ in labels))
    for a, b in combos:
        for n in params:
            print(f"{SHORT.get(a, a) + '→' + SHORT.get(b, b):<16}{n:<7}" + "".join(
                f"{get(r, a, b, n, 'r2'):>8.3f}{get(r, a, b, n, 'bias'):>+10.2f}{get(r, a, b, n, 'spread'):>12.2f}"
                for r in res))
    if len(set(noise)) == len(noise) > 1:
        plot_vs_noise(res, noise, params, combos, get, out_dir / "vs_noise.png")
    print(f"saved to {out_dir}/")


def plot_vs_noise(res, noise, params, combos, get, path):
    order = np.argsort(noise)
    x = np.array(noise, float)[order]
    xplot = np.where(x == 0, x[x > 0].min() / 2 if (x > 0).any() else 1, x)
    groups = {"self": [c for c in combos if c[0] == c[1]], "cross": [c for c in combos if c[0] != c[1]]}
    rows = [("r2", "R²"), ("absbias", "|median bias|\n(% of range)"), ("spread", "16–84% half-width\n(% of range)"),
            ("ratio", "|bias| / half-width")]

    def metric(r, a, b, n, key):
        if key == "absbias":
            return abs(get(r, a, b, n, "bias"))
        if key == "ratio":
            return abs(get(r, a, b, n, "bias")) / get(r, a, b, n, "spread")
        return get(r, a, b, n, key)

    fig, axes = plt.subplots(len(rows), len(params), figsize=(2.6 * len(params), 9), sharex=True)
    for j, n in enumerate(params):
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for g, (gname, pairs) in enumerate(groups.items()):
                vals = np.array([[metric(res[k], a, b, n, key) for k in order] for a, b in pairs])
                ax.fill_between(xplot, vals.min(0), vals.max(0), color=f"C{g}", alpha=0.15)
                ax.plot(xplot, vals.mean(0), "o-", ms=3, color=f"C{g}", label=f"{gname} (mean, band = range)")
            ax.set_xscale("log")
            ax.set_xticks(xplot, [f"{v:g}" for v in x])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(n)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("test noise (ppm)", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Noiseless-trained XGBoost retrieval vs test noise: self vs cross-code")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
