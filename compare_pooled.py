"""Compare a pooled retrieval model (trained on all codes) with the code-specific models.

For every tested code and parameter it shows R^2, the median bias and the 16-84% half-width of
predicted - true (bias and half-width as % of the prior range) for:
  - self:   the model trained on the tested code,
  - cross:  the models trained on the other codes (mean, with min/max),
  - pooled: the single model trained on all codes together.

Usage:
    python compare_pooled.py runs/retrieval runs/retrieval_pooled
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from anomaly import LABELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("per_code", help="ml_retrieval.py run with one model per code")
    p.add_argument("pooled", help="ml_retrieval.py --pooled run")
    p.add_argument("--out", default=None, help="default: the pooled run directory")
    return p.parse_args()


def main():
    args = parse_args()
    per = json.loads((Path(args.per_code) / "retrieval_results.json").read_text())
    pool = json.loads((Path(args.pooled) / "retrieval_results.json").read_text())
    params, rng = per["params"], per["prior_range"]
    codes = list(per["train_codes"])
    out = Path(args.out or args.pooled)

    def metric(t, key, n):
        if key == "r2":
            return t["r2"]
        return 100 * (t["median_bias"] if key == "bias" else t["spread_16_84"]) / rng[n]

    table = {}
    for b in codes:
        for n in params:
            self_t = per["train_codes"][b][n]["tested_on"][b]
            cross_t = [per["train_codes"][a][n]["tested_on"][b] for a in codes if a != b]
            pool_t = pool["train_codes"]["pooled"][n]["tested_on"][b]
            table[(b, n)] = {key: {"self": metric(self_t, key, n),
                                   "cross": [metric(t, key, n) for t in cross_t],
                                   "pooled": metric(pool_t, key, n)}
                             for key in ("r2", "bias", "spread")}
    (out / "pooled_vs_per_code.json").write_text(json.dumps(
        {f"{b}/{n}": v for (b, n), v in table.items()}, indent=2))

    print(f"{'tested on':<15}{'param':<7}{'R² self':>9}{'cross':>14}{'pooled':>8}"
          f"{'bias% self':>12}{'cross':>16}{'pooled':>8}")
    for b in codes:
        for n in params:
            t = table[(b, n)]
            print(f"{LABELS[b]:<15}{n:<7}{t['r2']['self']:>9.3f}"
                  f"{min(t['r2']['cross']):>7.3f}–{max(t['r2']['cross']):<6.3f}{t['r2']['pooled']:>8.3f}"
                  f"{t['bias']['self']:>+12.2f}{min(t['bias']['cross']):>+8.2f}…{max(t['bias']['cross']):<+7.2f}"
                  f"{t['bias']['pooled']:>+8.2f}")

    rows = [("r2", "R²"), ("bias", "median bias\n(% of range)"), ("spread", "16–84% half-width\n(% of range)")]
    fig, axes = plt.subplots(len(rows), len(params), figsize=(2.5 * len(params), 7.5), sharex=True)
    x = np.arange(len(codes))
    for j, n in enumerate(params):
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            selfv = [table[(b, n)][key]["self"] for b in codes]
            cross = [table[(b, n)][key]["cross"] for b in codes]
            poolv = [table[(b, n)][key]["pooled"] for b in codes]
            ax.scatter(x - 0.2, selfv, color="C0", s=18, label="self (own-code model)", zorder=3)
            ax.errorbar(x, [np.mean(c) for c in cross],
                        yerr=[[np.mean(c) - min(c) for c in cross], [max(c) - np.mean(c) for c in cross]],
                        fmt="s", color="C1", ms=4, capsize=2, label="cross (other-code models)", zorder=3)
            ax.scatter(x + 0.2, poolv, color="C2", marker="D", s=18, label="pooled (all codes)", zorder=3)
            if key == "bias":
                ax.axhline(0, color="grey", lw=0.8)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(n)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xticks(x, [LABELS[b] for b in codes], rotation=45, fontsize=6)
                ax.set_xlabel("tested on", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Pooled-code XGBoost retrieval vs code-specific models (self and cross)")
    fig.tight_layout()
    fig.savefig(out / "pooled_vs_per_code.png", dpi=130)
    plt.close(fig)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
