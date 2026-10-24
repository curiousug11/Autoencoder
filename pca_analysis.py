"""PCA on the preprocessed spectra to guide the choice of autoencoder bottleneck size.

Fits PCA on the training split (same cut / z-norm / split as train.py) and reports,
for k = 4..16 components, the explained variance and the test reconstruction MSE
in the same z-normalised units as the autoencoder loss.

Usage:
    python pca_analysis.py
    python pca_analysis.py --kmin 2 --kmax 20 --ae-metrics runs/ae_baseline/metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from data import DATA_DIR, prepare_splits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/pca")
    p.add_argument("--min-feature-height-ppm", type=float, default=210.0)
    p.add_argument("--kmin", type=int, default=4)
    p.add_argument("--kmax", type=int, default=16)
    p.add_argument("--ae-metrics", default="runs/ae_baseline/metrics.json",
                   help="optional autoencoder metrics.json to overlay on the MSE plot")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_pca(x):
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    explained_var = s**2 / (len(x) - 1)
    return mean, vt, explained_var


def pca_reconstruct(x, mean, vt, k):
    z = (x - mean) @ vt[:k].T
    return z @ vt[:k] + mean


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits, _, data_info = prepare_splits(args.data_dir, args.min_feature_height_ppm, seed=args.seed)
    x_train = splits["train"].spectra.astype(np.float64)
    x_test = splits["test"].spectra.astype(np.float64)

    mean, vt, explained_var = fit_pca(x_train)
    ratio = explained_var / explained_var.sum()
    cumulative = np.cumsum(ratio)

    rows = []
    for k in range(args.kmin, args.kmax + 1):
        err = ((pca_reconstruct(x_test, mean, vt, k) - x_test) ** 2).mean(axis=1)
        rows.append({
            "k": k,
            "explained_variance_ratio": float(ratio[k - 1]),
            "cumulative_explained_variance": float(cumulative[k - 1]),
            "test_mse": float(err.mean()),
            "test_mse_median": float(np.median(err)),
            "test_mse_p95": float(np.percentile(err, 95)),
        })

    thresholds = {f"{t:g}": int(np.searchsorted(cumulative, t) + 1)
                  for t in (0.99, 0.999, 0.9999)}
    results = {**data_info, "components_for_cumulative_variance": thresholds, "per_k": rows}
    (out / "pca_results.json").write_text(json.dumps(results, indent=2))

    print(f"components needed for cumulative variance: {thresholds}")
    print(f"{'k':>3} {'var ratio':>10} {'cumulative':>11} {'1-cumul':>9} {'test MSE':>10} {'median':>10} {'p95':>10}")
    for r in rows:
        print(f"{r['k']:>3} {r['explained_variance_ratio']:>10.2e} {r['cumulative_explained_variance']:>11.6f} "
              f"{1 - r['cumulative_explained_variance']:>9.2e} {r['test_mse']:>10.2e} "
              f"{r['test_mse_median']:>10.2e} {r['test_mse_p95']:>10.2e}")

    ae = None
    ae_path = Path(args.ae_metrics)
    if ae_path.exists():
        m = json.loads(ae_path.read_text())
        ae = (m["args"]["latent"], m["test_mse"])

    ks = np.array([r["k"] for r in rows])
    all_k = np.arange(1, len(ratio) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.semilogy(all_k, ratio, "k.-", lw=1)
    ax.axvspan(args.kmin, args.kmax, color="C0", alpha=0.1)
    ax.set(xlabel="component", ylabel="explained variance ratio", title="Scree plot")

    ax = axes[1]
    ax.semilogy(all_k, 1 - cumulative + 1e-16, "k.-", lw=1)
    for t, k in thresholds.items():
        ax.axhline(1 - float(t), color="grey", ls=":", lw=1)
        ax.text(len(ratio), 1 - float(t), f"{float(t) * 100:g}% (k={k})", ha="right", va="bottom", fontsize=8)
    ax.axvspan(args.kmin, args.kmax, color="C0", alpha=0.1)
    ax.set(xlabel="number of components k", ylabel="unexplained variance (1 - cumulative)",
           title="Cumulative explained variance")

    ax = axes[2]
    ax.semilogy(ks, [r["test_mse"] for r in rows], "o-", label="PCA test MSE (mean)")
    ax.semilogy(ks, [r["test_mse_median"] for r in rows], "s--", ms=4, label="PCA test MSE (median)")
    if ae:
        ax.semilogy([ae[0]], [ae[1]], "r*", ms=14, label=f"AE (latent={ae[0]}) test MSE")
    ax.set(xlabel="number of components k", ylabel="reconstruction MSE (z-normalised)",
           title="Test reconstruction error", xticks=ks)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "pca_bottleneck.png", dpi=150)
    plt.close(fig)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
