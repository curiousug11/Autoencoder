"""Score noisy spectra against their known error bars (chi-square) instead of z-normalised MSE.

For an observed spectrum y with per-bin errors sigma (ppm):
  1. z-normalise y with its own mean m and std s, reconstruct with the code's autoencoder,
  2. map the reconstruction back to transit depth: y_hat = AE(z) * s + m,
  3. score chi2_nu = sum(((y - y_hat) / sigma)^2) / n_bins.
A spectrum the code can produce should score chi2_nu ~ 1 whatever the noise level.

Two thresholds, both aiming at a false-positive rate of --fpr:
  - theoretical: the (1 - fpr) quantile of a chi-square with n_bins degrees of freedom,
  - empirical: the (1 - fpr) quantile of chi2_nu on noisy own-code *validation* spectra
    (simulated at the same noise level), which absorbs the model's own reconstruction error.

Model sets compared on the same noisy test spectra:
  - "noiseless": the anomaly models trained on clean spectra (--runs),
  - "denoising": models trained with noise at that level (--denoise-template, if present),
  - "oracle": chi2 against the training code's true noiseless spectrum of the same planet,
    the best any method could do if it knew the planet's parameters exactly.

Usage:
    python chi2_score.py
    python chi2_score.py --sigmas 10 20 30 50 --denoise-template runs/anomaly_noise/sigma{sigma:g}
    python chi2_score.py --sigma-file errors_ppm.npy   # per-bin error shape, scaled by each sigma
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

KEYS = ["auc_chi2", "auc_mse", "tpr_theory", "tpr_empirical", "fpr_theory", "fpr_empirical", "median_chi2nu"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--runs", default="runs/anomaly", help="noiseless-trained models")
    p.add_argument("--denoise-template", default="runs/anomaly_noise/sigma{sigma:g}",
                   help="directory of models trained at noise sigma; skipped where missing")
    p.add_argument("--out", default="runs/anomaly/chi2")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--seed", type=int, default=42, help="split seed used by anomaly.py")
    p.add_argument("--sigmas", type=float, nargs="+", default=[2, 5, 10, 20, 30, 50, 100],
                   help="noise per bin in ppm (multiplies --sigma-file if given)")
    p.add_argument("--sigma-file", default=None,
                   help=".npy with relative per-bin errors (n_bins,), normalised to mean 1")
    p.add_argument("--repeats", type=int, default=2, help="noise realisations per sigma")
    p.add_argument("--fpr", type=float, default=0.01, help="target false-positive rate for the thresholds")
    return p.parse_args()


@torch.no_grad()
def reconstruct_depth(model, y):
    """AE reconstruction of raw spectra y, in transit-depth units, plus z-space input/output."""
    z, m, s = znormalise(y)
    z_hat = model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64)
    return z_hat * s[:, None] + m[:, None], z, z_hat


def scores(model, y, err):
    y_hat, z, z_hat = reconstruct_depth(model, y)
    return (((y - y_hat) / err) ** 2).mean(1), ((z - z_hat) ** 2).mean(1)


def model_sets(args, sigma):
    sets = {"noiseless": args.runs}
    d = Path(args.denoise_template.format(sigma=sigma))
    if all((d / f"train_{c}" / "metrics.json").exists() for c in args.datasets):
        sets["denoising"] = d
    return sets


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    splits = prepare_splits(args.data_dir, seed=args.seed, dataset=args.datasets[0], cut_datasets=args.datasets)[0]
    raw = {d: load_raw(args.data_dir, d)[0] for d in args.datasets}
    clean = {d: raw[d][splits["test"].index] for d in args.datasets}
    clean_val = {d: raw[d][splits["val"].index] for d in args.datasets}
    n_bins = clean[args.datasets[0]].shape[1]
    shape = np.ones(n_bins)
    if args.sigma_file:
        shape = np.load(args.sigma_file).astype(float)
        shape /= shape.mean()
    theory_thr = float(np.quantile(rng.chisquare(n_bins, 2_000_000), 1 - args.fpr)) / n_bins
    pairs = [f"{a}->{b}" for a in args.datasets for b in args.datasets if b != a]

    results = {}  # results[set][pair or "normal:<code>"][key] -> list over sigmas (None where missing)
    examples = {}
    for si, sigma in enumerate(args.sigmas):
        err = sigma * 1e-6 * shape
        sets = model_sets(args, sigma)
        acc = {}
        for rep in range(args.repeats):
            obs = {d: clean[d] + rng.normal(0, 1, clean[d].shape) * err for d in args.datasets}
            obs_val = {d: clean_val[d] + rng.normal(0, 1, clean_val[d].shape) * err for d in args.datasets}
            for name, run_root in sets.items():
                for a in args.datasets:
                    model = load_model(select_run(run_root, a)[0] / "model.pt")
                    chi2, mse = {}, {}
                    for d in args.datasets:
                        chi2[d], mse[d] = scores(model, obs[d], err)
                    emp_thr = float(np.quantile(scores(model, obs_val[a], err)[0], 1 - args.fpr))
                    nk = acc.setdefault(name, {}).setdefault(f"normal:{a}", {})
                    nk.setdefault("fpr_theory", []).append(float((chi2[a] > theory_thr).mean()))
                    nk.setdefault("fpr_empirical", []).append(float((chi2[a] > emp_thr).mean()))
                    nk.setdefault("median_chi2nu", []).append(float(np.median(chi2[a])))
                    for b in args.datasets:
                        if b == a:
                            continue
                        pk = acc[name].setdefault(f"{a}->{b}", {})
                        pk.setdefault("auc_chi2", []).append(roc_auc(chi2[a], chi2[b]))
                        pk.setdefault("auc_mse", []).append(roc_auc(mse[a], mse[b]))
                        pk.setdefault("tpr_theory", []).append(float((chi2[b] > theory_thr).mean()))
                        pk.setdefault("tpr_empirical", []).append(float((chi2[b] > emp_thr).mean()))
                    if rep == 0:
                        examples[(name, sigma, a)] = (chi2, emp_thr)
            # Oracle: chi2 against the training code's true noiseless spectrum.
            for a in args.datasets:
                oracle = {d: (((obs[d] - clean[a]) / err) ** 2).mean(1) for d in args.datasets}
                nk = acc.setdefault("oracle", {}).setdefault(f"normal:{a}", {})
                nk.setdefault("fpr_theory", []).append(float((oracle[a] > theory_thr).mean()))
                nk.setdefault("median_chi2nu", []).append(float(np.median(oracle[a])))
                for b in args.datasets:
                    if b != a:
                        pk = acc["oracle"].setdefault(f"{a}->{b}", {})
                        pk.setdefault("auc_chi2", []).append(roc_auc(oracle[a], oracle[b]))
                        pk.setdefault("tpr_theory", []).append(float((oracle[b] > theory_thr).mean()))
        for name, rows in acc.items():
            for row, vals in rows.items():
                dst = results.setdefault(name, {}).setdefault(row, {})
                for k, v in vals.items():
                    dst.setdefault(k, [None] * len(args.sigmas))[si] = float(np.mean(v))
        report(args, sigma, si, results, theory_thr, pairs)

    summary = {"sigmas_ppm": args.sigmas, "sigma_file": args.sigma_file, "theory_threshold_chi2nu": theory_thr,
               "nominal_fpr": args.fpr, "n_bins": n_bins, "n_test_planets": int(len(splits["test"].index)),
               "results": results}
    (out / "chi2_results.json").write_text(json.dumps(summary, indent=2))
    plot_summary(args, results, pairs, out / "chi2_auc_vs_noise.png")
    plot_hists(args, examples, theory_thr, out / "chi2_histograms.png")
    print(f"\nsaved to {out}/")


def mean_over(results, name, rows, key, si):
    vals = [results[name][r][key][si] for r in rows if key in results[name].get(r, {})]
    vals = [v for v in vals if v is not None]
    return (np.mean(vals), np.min(vals), np.max(vals)) if vals else (np.nan, np.nan, np.nan)


def report(args, sigma, si, results, theory_thr, pairs):
    normals = [f"normal:{a}" for a in args.datasets]
    print(f"\nsigma {sigma:g} ppm   (theoretical threshold chi2_nu > {theory_thr:.3f})")
    print(f"  {'model set':<11}{'AUC chi2':>10}{'AUC mse':>9}{'med chi2nu':>11}"
          f"{'FPR theo':>10}{'TPR theo':>10}{'FPR emp':>9}{'TPR emp':>9}")
    for name in ("noiseless", "denoising", "oracle"):
        if name not in results or results[name][pairs[0]]["auc_chi2"][si] is None:
            continue
        f = lambda key, rows: mean_over(results, name, rows, key, si)[0]  # noqa: E731
        print(f"  {name:<11}{f('auc_chi2', pairs):>10.3f}{f('auc_mse', pairs):>9.3f}"
              f"{f('median_chi2nu', normals):>11.2f}{f('fpr_theory', normals):>10.1%}{f('tpr_theory', pairs):>10.1%}"
              f"{f('fpr_empirical', normals):>9.1%}{f('tpr_empirical', pairs):>9.1%}".replace("nan%", "  - ").replace(" nan", "  - "))


def plot_summary(args, results, pairs, path):
    x = np.array(args.sigmas)
    normals = [f"normal:{a}" for a in args.datasets]
    styles = {"oracle": ("oracle (true params known)", "k", ":"),
              "denoising": ("denoising AE, χ²", "C2", "-"),
              "noiseless": ("noiseless AE, χ²", "C3", "-")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [(axes[0], "auc_chi2", pairs, "AUC (mean; band = range over 6 pairs)", "Separation of codes"),
              (axes[1], "tpr_empirical", pairs, "other-code spectra flagged",
               f"Detection rate at {args.fpr:.0%} FPR (threshold from noisy val)"),
              (axes[2], "fpr_theory", normals, "own-code spectra flagged",
               f"FPR at the theoretical χ² {1 - args.fpr:.0%} threshold")]
    for ax, key, rows, ylabel, title in panels:
        for name, (label, c, ls) in styles.items():
            if name not in results:
                continue
            k = "tpr_theory" if (name == "oracle" and key == "tpr_empirical") else key
            stats = np.array([mean_over(results, name, rows, k, i) for i in range(len(x))])
            ok = ~np.isnan(stats[:, 0])
            if not ok.any():
                continue
            ax.fill_between(x[ok], stats[ok, 1], stats[ok, 2], color=c, alpha=0.15)
            ax.plot(x[ok], stats[ok, 0], color=c, ls=ls, marker="o", ms=4, label=label)
        if key == "auc_mse" or key == "auc_chi2":
            stats = np.array([mean_over(results, "noiseless", pairs, "auc_mse", i) for i in range(len(x))])
            ax.plot(x, stats[:, 0], color="C0", ls="--", marker="o", ms=3, label="noiseless AE, z-norm MSE")
            ax.axhline(0.5, color="grey", lw=0.8)
            ax.set_ylim(0.45, 1.0)
        if key == "fpr_theory":
            ax.axhline(args.fpr, color="grey", lw=0.8, ls="--", label="nominal")
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set(xscale="log", xlabel="noise per bin σ (ppm)", ylabel=ylabel, title=title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hists(args, examples, theory_thr, path):
    name = "denoising" if any(k[0] == "denoising" for k in examples) else "noiseless"
    sigmas = [s for s in args.sigmas if (name, s, args.datasets[0]) in examples and s in (10, 20, 50)]
    sigmas = sigmas or [s for s in args.sigmas if (name, s, args.datasets[0]) in examples][:3]
    fig, axes = plt.subplots(len(sigmas), len(args.datasets), figsize=(5.5 * len(args.datasets), 3.2 * len(sigmas)),
                             squeeze=False)
    for i, s in enumerate(sigmas):
        for j, a in enumerate(args.datasets):
            ax = axes[i, j]
            chi2, emp_thr = examples[(name, s, a)]
            allv = np.log10(np.concatenate(list(chi2.values())))
            bins = np.linspace(np.percentile(allv, 0.1), np.percentile(allv, 99.9), 70)
            for k, d in enumerate(args.datasets):
                normal = d == a
                ax.hist(np.log10(chi2[d]), bins=bins, histtype="stepfilled" if normal else "step",
                        alpha=0.35 if normal else 1, lw=1.6, color=f"C{k}",
                        label=f"{LABELS[d]} ({'normal' if normal else 'other code'})")
            ax.axvline(np.log10(emp_thr), color="k", ls="--", lw=1, label=f"{1 - args.fpr:.0%} of noisy val")
            ax.axvline(np.log10(theory_thr), color="grey", ls=":", lw=1, label=f"χ² {1 - args.fpr:.0%} (theory)")
            ax.set_title(f"{name} AE trained on {LABELS[a]}, σ = {s:g} ppm", fontsize=10)
            if i == len(sigmas) - 1:
                ax.set_xlabel("log10 reduced χ² vs error bars")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
