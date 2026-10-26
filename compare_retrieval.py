"""Compare two ml_retrieval.py runs, e.g. the same models tested on noiseless vs noisy spectra.

For every (training code, tested code, parameter) it shows R^2, the median bias and the 16-84%
half-width of predicted - true, the last two as % of the parameter's prior range.

Usage:
    python compare_retrieval.py runs/retrieval runs/retrieval_test10ppm --labels "0 ppm" "10 ppm"
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from anomaly import LABELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT = {"pos": "POS", "prt": "pRT", "taurex": "TRx"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="ml_retrieval.py output directories")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default=None, help="default: <last run>/comparison.png")
    return p.parse_args()


def main():
    args = parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    res = [json.loads((Path(r) / "retrieval_results.json").read_text()) for r in args.runs]
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
    out = Path(args.out) if args.out else Path(args.runs[-1]) / "comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print(f"{'trained→tested':<16}{'param':<7}" + "".join(f"{lab:>30}" for lab in labels))
    print(f"{'':<23}" + "".join(f"{'R²':>8}{'bias%':>10}{'spread%':>12}" for _ in labels))
    for a, b in combos:
        for n in params:
            print(f"{SHORT.get(a, a) + '→' + SHORT.get(b, b):<16}{n:<7}" + "".join(
                f"{get(r, a, b, n, 'r2'):>8.3f}{get(r, a, b, n, 'bias'):>+10.2f}{get(r, a, b, n, 'spread'):>12.2f}"
                for r in res))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
