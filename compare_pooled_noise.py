"""Pooled vs code-specific retrieval as a function of test noise.

Reads pairs of ml_retrieval.py runs (code-specific and --pooled) tested at the same noise levels,
and plots R^2, |median bias| and the 16-84% half-width (bias and half-width as % of the prior
range) against the noise level for:
  - self:   model trained on the tested code (mean over the 3 codes, band = range),
  - cross:  models trained on another code (mean over the 6 pairs, band = range),
  - pooled: one model trained on all codes (mean over the 3 tested codes, band = range).

Usage:
    python compare_pooled_noise.py --noise 0 10 20 30 50 \
        --per-code runs/retrieval runs/retrieval_test{10,20,30,50}ppm \
        --pooled runs/retrieval_pooled runs/retrieval_pooled_test{10,20,30,50}ppm \
        --out runs/retrieval_pooled_noise
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--noise", type=float, nargs="+", required=True)
    p.add_argument("--per-code", nargs="+", required=True)
    p.add_argument("--pooled", nargs="+", required=True)
    p.add_argument("--out", default="runs/retrieval_pooled_noise")
    return p.parse_args()


def load(d):
    return json.loads((Path(d) / "retrieval_results.json").read_text())


def main():
    args = parse_args()
    assert len(args.noise) == len(args.per_code) == len(args.pooled)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = [load(d) for d in args.per_code]
    pool = [load(d) for d in args.pooled]
    params, rng = per[0]["params"], per[0]["prior_range"]
    codes = list(per[0]["train_codes"])

    def m(t, key, n):
        if key == "r2":
            return t["r2"]
        if key == "absbias":
            return 100 * abs(t["median_bias"]) / rng[n]
        if key == "bias":
            return 100 * t["median_bias"] / rng[n]
        return 100 * t["spread_16_84"] / rng[n]

    def groups(k, n, key):
        return {
            "self": [m(per[k]["train_codes"][b][n]["tested_on"][b], key, n) for b in codes],
            "cross": [m(per[k]["train_codes"][a][n]["tested_on"][b], key, n) for a in codes for b in codes if a != b],
            "pooled": [m(pool[k]["train_codes"]["pooled"][n]["tested_on"][b], key, n) for b in codes],
        }

    summary = {f"{s:g}": {n: {key: groups(k, n, key) for key in ("r2", "bias", "spread")} for n in params}
               for k, s in enumerate(args.noise)}
    (out / "pooled_vs_noise.json").write_text(json.dumps(summary, indent=2))

    print("mean R²  self / cross / pooled")
    print(f"{'param':<7}" + "".join(f"{f'{s:g} ppm':>22}" for s in args.noise))
    for n in params:
        row = f"{n:<7}"
        for k in range(len(args.noise)):
            g = groups(k, n, "r2")
            row += f"{np.mean(g['self']):>8.3f}{np.mean(g['cross']):>7.3f}{np.mean(g['pooled']):>7.3f}"
        print(row)
    print("\npooled model: median bias (% of range) on POS / petitRADTRANS / TauREx test spectra")
    for n in params:
        print(f"{n:<7}" + "".join(
            "  " + "/".join(f"{v:+.2f}" for v in groups(k, n, "bias")["pooled"]) for k in range(len(args.noise))))

    x = np.array(args.noise, float)
    xplot = np.where(x == 0, x[x > 0].min() / 2 if (x > 0).any() else 1, x)
    rows = [("r2", "R²"), ("absbias", "|median bias|\n(% of range)"), ("spread", "16–84% half-width\n(% of range)")]
    colors = {"self": "C0", "cross": "C1", "pooled": "C2"}
    fig, axes = plt.subplots(len(rows), len(params), figsize=(2.6 * len(params), 7.5), sharex=True)
    for j, n in enumerate(params):
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for g in ("self", "cross", "pooled"):
                vals = np.array([groups(k, n, key)[g] for k in range(len(args.noise))])
                ax.fill_between(xplot, vals.min(1), vals.max(1), color=colors[g], alpha=0.15)
                ax.plot(xplot, vals.mean(1), "o-", ms=3, color=colors[g], label=g)
            ax.set_xscale("log")
            ax.set_xticks(xplot, [f"{v:g}" for v in x])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(n)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("test noise (ppm)", fontsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Noiseless-trained XGBoost retrieval vs test noise: self, cross and pooled-code models "
                 "(line = mean, band = range)")
    fig.tight_layout()
    fig.savefig(out / "pooled_vs_noise.png", dpi=130)
    plt.close(fig)

    # Signed bias of the pooled model per tested code, next to self and cross, for every parameter.
    fig, axes = plt.subplots(len(codes), len(params), figsize=(2.6 * len(params), 7), sharex=True)
    for i, b in enumerate(codes):
        for j, n in enumerate(params):
            ax = axes[i, j]
            selfb = [m(per[k]["train_codes"][b][n]["tested_on"][b], "bias", n) for k in range(len(x))]
            ax.plot(xplot, selfb, "o-", ms=3, color="C0", label="self")
            for a in codes:
                if a != b:
                    cb = [m(per[k]["train_codes"][a][n]["tested_on"][b], "bias", n) for k in range(len(x))]
                    ax.plot(xplot, cb, "s--", ms=2.5, lw=1, color="C1", label=f"cross (trained {LABELS[a]})")
            pb = [m(pool[k]["train_codes"]["pooled"][n]["tested_on"][b], "bias", n) for k in range(len(x))]
            ax.plot(xplot, pb, "D-", ms=3, color="C2", label="pooled")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(xplot, [f"{v:g}" for v in x])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(n)
            if j == 0:
                ax.set_ylabel(f"tested on {LABELS[b]}\nbias (% of range)", fontsize=7)
            if i == len(codes) - 1:
                ax.set_xlabel("test noise (ppm)", fontsize=7)
    axes[0, 0].legend(fontsize=5)
    fig.suptitle("Median bias (predicted − true) vs test noise, per tested code")
    fig.tight_layout()
    fig.savefig(out / "pooled_bias_vs_noise.png", dpi=130)
    plt.close(fig)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
