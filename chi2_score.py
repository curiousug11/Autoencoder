"""Score noisy spectra against their known error bars (chi-square) instead of z-normalised MSE.

For an observed spectrum y with per-bin errors sigma (ppm):
  1. z-normalise y with its own mean m and std s, reconstruct with the code's autoencoder,
  2. map the reconstruction back to transit depth: y_hat = AE(z) * s + m,
  3. score chi2_nu = sum(((y - y_hat) / sigma)^2) / n_bins.
A spectrum the code can produce should score chi2_nu ~ 1 whatever the noise level, so one
threshold works everywhere. Here the threshold is the 99th percentile of a chi-square with n_bins
degrees of freedom: about 1% of normal spectra should be flagged if the model were perfect.

The models are the noiseless-trained anomaly models (anomaly.py); they are not retrained.
For comparison the script also reports:
  - the old score (z-normalised MSE) on the same noisy spectra,
  - an "oracle" chi2 against the training code's true noiseless spectrum of the same planet,
    the best any method could do if it knew the planet's parameters exactly.

Usage:
    python chi2_score.py
    python chi2_score.py --sigmas 10 30 100 --sigma-file errors_ppm.npy   # per-bin errors, scaled
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from anomaly import LABELS, load_model, roc_auc, select_run
from data import DATA_DIR, DATASETS, load_raw, prepare_splits, znormalise

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--runs", default="runs/anomaly")
    p.add_argument("--out", default="runs/anomaly/chi2")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--seed", type=int, default=42, help="split seed used by anomaly.py")
    p.add_argument("--sigmas", type=float, nargs="+", default=[2, 5, 10, 20, 30, 50, 100],
                   help="noise per bin in ppm (multiplies --sigma-file if given)")
    p.add_argument("--sigma-file", default=None,
                   help=".npy with relative per-bin errors (n_bins,), normalised to mean 1")
    p.add_argument("--repeats", type=int, default=2, help="noise realisations per sigma")
    p.add_argument("--fpr", type=float, default=0.01, help="target false-positive rate for the threshold")
    return p.parse_args()


@torch.no_grad()
def reconstruct_depth(model, y):
    """AE reconstruction of raw spectra y, in transit-depth units."""
    z, m, s = znormalise(y)
    z_hat = model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64)
    return z_hat * s[:, None] + m[:, None], z, z_hat


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    test_index = prepare_splits(args.data_dir, seed=args.seed, dataset=args.datasets[0],
                                cut_datasets=args.datasets)[0]["test"].index
    clean = {d: load_raw(args.data_dir, d)[0][test_index] for d in args.datasets}
    n_bins = clean[args.datasets[0]].shape[1]
    models = {d: load_model(select_run(args.runs, d)[0] / "model.pt") for d in args.datasets}
    shape = np.ones(n_bins)
    if args.sigma_file:
        shape = np.load(args.sigma_file).astype(float)
        shape /= shape.mean()
    threshold = float(np.quantile(rng.chisquare(n_bins, 2_000_000), 1 - args.fpr)) / n_bins

    pairs = [(a, b) for a in args.datasets for b in args.datasets if b != a]
    keys = ["auc_chi2", "auc_mse", "auc_oracle", "tpr_chi2", "tpr_oracle"]
    res = {f"{a}->{b}": {k: [] for k in keys} for a, b in pairs}
    normal_stats = {a: {"fpr_chi2": [], "median_chi2nu": [], "fpr_oracle": []} for a in args.datasets}
    examples = {}

    for sigma in args.sigmas:
        err = sigma * 1e-6 * shape
        acc = {p: {k: [] for k in keys} for p in res}
        nacc = {a: {k: [] for k in normal_stats[a]} for a in args.datasets}
        for rep in range(args.repeats):
            obs = {d: clean[d] + rng.normal(0, 1, clean[d].shape) * err for d in args.datasets}
            for a in args.datasets:
                chi2, mse, oracle = {}, {}, {}
                for d in args.datasets:
                    y_hat, z, z_hat = reconstruct_depth(models[a], obs[d])
                    chi2[d] = (((obs[d] - y_hat) / err) ** 2).mean(1)
                    mse[d] = ((z - z_hat) ** 2).mean(1)
                    oracle[d] = (((obs[d] - clean[a]) / err) ** 2).mean(1)
                nacc[a]["fpr_chi2"].append(float((chi2[a] > threshold).mean()))
                nacc[a]["fpr_oracle"].append(float((oracle[a] > threshold).mean()))
                nacc[a]["median_chi2nu"].append(float(np.median(chi2[a])))
                for b in args.datasets:
                    if b == a:
                        continue
                    p = f"{a}->{b}"
                    acc[p]["auc_chi2"].append(roc_auc(chi2[a], chi2[b]))
                    acc[p]["auc_mse"].append(roc_auc(mse[a], mse[b]))
                    acc[p]["auc_oracle"].append(roc_auc(oracle[a], oracle[b]))
                    acc[p]["tpr_chi2"].append(float((chi2[b] > threshold).mean()))
                    acc[p]["tpr_oracle"].append(float((oracle[b] > threshold).mean()))
                if rep == 0:
                    examples[(sigma, a)] = chi2
        for p in res:
            for k in keys:
                res[p][k].append(float(np.mean(acc[p][k])))
        for a in args.datasets:
            for k in normal_stats[a]:
                normal_stats[a][k].append(float(np.mean(nacc[a][k])))

        print(f"\nsigma {sigma:g} ppm (threshold chi2_nu > {threshold:.3f}, nominal FPR {args.fpr:.0%})")
        for a in args.datasets:
            n = normal_stats[a]
            print(f"  {LABELS[a]:<14} normal: median chi2_nu {n['median_chi2nu'][-1]:.2f}, "
                  f"flagged {n['fpr_chi2'][-1]:.1%} (oracle {n['fpr_oracle'][-1]:.1%})")
        print(f"  {'pair':<16}{'AUC chi2':>9}{'AUC mse':>9}{'AUC oracle':>11}{'TPR chi2':>10}{'TPR oracle':>11}")
        for p in res:
            r = res[p]
            print(f"  {p:<16}{r['auc_chi2'][-1]:>9.3f}{r['auc_mse'][-1]:>9.3f}{r['auc_oracle'][-1]:>11.3f}"
                  f"{r['tpr_chi2'][-1]:>10.1%}{r['tpr_oracle'][-1]:>11.1%}")

    summary = {"sigmas_ppm": args.sigmas, "sigma_file": args.sigma_file, "threshold_chi2nu": threshold,
               "nominal_fpr": args.fpr, "n_bins": n_bins, "n_test_planets": int(len(test_index)),
               "pairs": res, "normal": normal_stats}
    (out / "chi2_results.json").write_text(json.dumps(summary, indent=2))
    plot_auc(args, res, normal_stats, out / "chi2_auc_vs_noise.png")
    plot_hists(args, examples, threshold, out / "chi2_histograms.png")
    print(f"\nsaved to {out}/")


def plot_auc(args, res, normal_stats, path):
    x = np.array(args.sigmas)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    styles = {"auc_oracle": ("oracle chi2 (true params known)", "k", ":"),
              "auc_chi2": ("AE chi2 vs error bars", "C3", "-"),
              "auc_mse": ("AE z-norm MSE (old score)", "C0", "--")}
    for k, (label, c, ls) in styles.items():
        vals = np.array([res[p][k] for p in res])
        axes[0].fill_between(x, vals.min(0), vals.max(0), color=c, alpha=0.15)
        axes[0].plot(x, vals.mean(0), color=c, ls=ls, marker="o", ms=4, label=label)
    axes[0].axhline(0.5, color="grey", lw=0.8)
    axes[0].set(xscale="log", xlabel="noise per bin σ (ppm)", ylabel="AUC (mean, band = range over 6 pairs)",
                title="Separation of codes", ylim=(0.45, 1.0))
    axes[0].legend(fontsize=8)

    for k, (label, c, ls) in [("tpr_oracle", styles["auc_oracle"]), ("tpr_chi2", styles["auc_chi2"])]:
        vals = np.array([res[p][k] for p in res])
        axes[1].fill_between(x, vals.min(0), vals.max(0), color=c, alpha=0.15)
        axes[1].plot(x, vals.mean(0), color=c, ls=ls, marker="o", ms=4, label=label.split(" (")[0])
    axes[1].set(xscale="log", xlabel="noise per bin σ (ppm)", ylabel="other-code spectra flagged",
                title=f"Detection rate at the χ² {1 - args.fpr:.0%} threshold", ylim=(0, 1))
    axes[1].legend(fontsize=8)

    for i, a in enumerate(args.datasets):
        axes[2].plot(x, normal_stats[a]["fpr_chi2"], "o-", color=f"C{i}", ms=4, label=f"{LABELS[a]} (AE)")
    axes[2].plot(x, np.mean([normal_stats[a]["fpr_oracle"] for a in args.datasets], 0), "k:", label="oracle")
    axes[2].axhline(args.fpr, color="grey", lw=0.8, ls="--", label="nominal")
    axes[2].set(xscale="log", yscale="log", xlabel="noise per bin σ (ppm)", ylabel="own-code spectra flagged",
                title="False-positive rate (calibration)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hists(args, examples, threshold, path):
    sigmas = [s for s in args.sigmas if s in (5, 20, 50)] or args.sigmas[:3]
    fig, axes = plt.subplots(len(sigmas), len(args.datasets), figsize=(5.5 * len(args.datasets), 3.2 * len(sigmas)),
                             squeeze=False)
    for i, s in enumerate(sigmas):
        for j, a in enumerate(args.datasets):
            ax = axes[i, j]
            chi2 = examples[(s, a)]
            allv = np.log10(np.concatenate(list(chi2.values())))
            bins = np.linspace(np.percentile(allv, 0.1), np.percentile(allv, 99.9), 70)
            for k, d in enumerate(args.datasets):
                normal = d == a
                ax.hist(np.log10(chi2[d]), bins=bins, histtype="stepfilled" if normal else "step",
                        alpha=0.35 if normal else 1, lw=1.6, color=f"C{k}",
                        label=f"{LABELS[d]} ({'normal' if normal else 'other code'})")
            ax.axvline(np.log10(threshold), color="k", ls="--", lw=1, label=f"χ² {1 - args.fpr:.0%} threshold")
            ax.set_title(f"AE trained on {LABELS[a]}, σ = {s:g} ppm", fontsize=10)
            if i == len(sigmas) - 1:
                ax.set_xlabel("log10 reduced χ² vs error bars")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
