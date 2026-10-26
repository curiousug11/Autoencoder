"""Which planet parameters make a code's own spectra hard to reconstruct?

For each code, take the selected anomaly-experiment model (lowest own-code val loss, see
anomaly.py) and score that code's own held-out spectra (val + test planets, never trained on).
The top --top-frac by reconstruction MSE are "poorly reconstructed". The script compares their
parameter distributions with the rest, and checks whether the same planets are hard in every code.

Usage:
    python poor_recon.py
    python poor_recon.py --top-frac 0.1 --param-names Rs Rp Mp T H2O CH4 CO CO2 NH3

Besides the 9 parameters (Rs, Rp, Mp, T, then 5 log abundances) it also checks derived
quantities: log10 surface gravity Mp/Rp^2 (Jupiter units), log10 T*Rp/Mp (proportional to
scale height over planet radius, H/Rp, at fixed mean molecular weight) and the feature height.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from anomaly import LABELS, load_model, per_spectrum_mse, select_run
from data import DATA_DIR, DATASETS, feature_height, load_raw, prepare_splits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--runs", default="runs/anomaly", help="anomaly.py output directory with trained models")
    p.add_argument("--out", default="runs/anomaly/poor_recon")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--seed", type=int, default=42, help="split seed used by anomaly.py")
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--param-names", nargs=9,
                   default=["Rs", "Rp", "Mp", "T", "chem1", "chem2", "chem3", "chem4", "chem5"])
    return p.parse_args()


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def auc(neg, pos):
    """P(value for a random poorly reconstructed planet > value for a random other planet)."""
    r = rank(np.concatenate([neg, pos])) + 1
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


DERIVED = ["log_g", "log_H_over_Rp", "feature_height_ppm"]


def features(params, fh):
    """Parameters plus derived columns (DERIVED). Columns 1-3 of params are Rp, Mp, T."""
    rp, mp, t = params[:, 1], params[:, 2], params[:, 3]
    return np.column_stack([params, np.log10(mp / rp**2), np.log10(t * rp / mp), fh])


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(args.param_names)

    # Same planets and split for every code (anomaly.py setup); use val + test planets.
    errors, params, index = {}, None, None
    for d in args.datasets:
        splits = prepare_splits(args.data_dir, seed=args.seed, dataset=d, cut_datasets=args.datasets)[0]
        x = np.concatenate([splits["val"].spectra, splits["test"].spectra])
        idx = np.concatenate([splits["val"].index, splits["test"].index])
        assert index is None or np.array_equal(idx, index)
        index = idx
        params = np.concatenate([splits["val"].params, splits["test"].params])
        run, _ = select_run(args.runs, d)
        errors[d] = per_spectrum_mse(load_model(run / "model.pt"), x)

    # Feature height (ppm) of each code's own raw spectrum, as an extra explanatory variable.
    fh = {d: feature_height(load_raw(args.data_dir, d)[0][index]) * 1e6 for d in args.datasets}
    columns = names + DERIVED

    n_top = int(round(args.top_frac * len(index)))
    poor = {d: np.argsort(errors[d])[::-1][:n_top] for d in args.datasets}
    results = {"n_planets": int(len(index)), "top_frac": args.top_frac, "n_poor": n_top, "codes": {}}
    for d in args.datasets:
        is_poor = np.zeros(len(index), bool)
        is_poor[poor[d]] = True
        feats = features(params, fh[d])
        res = {}
        for j, c in enumerate(columns):
            v = feats[:, j]
            res[c] = {
                "spearman_logmse": spearman(v, np.log10(errors[d])),
                "auc_poor_vs_rest": auc(v[~is_poor], v[is_poor]),
                "median_poor": float(np.median(v[is_poor])),
                "median_rest": float(np.median(v[~is_poor])),
            }
        results["codes"][d] = {"mse_threshold": float(np.sort(errors[d])[::-1][n_top - 1]), "params": res}

    overlap = {}
    for i, a in enumerate(args.datasets):
        for b in args.datasets[i + 1:]:
            inter = len(np.intersect1d(poor[a], poor[b]))
            overlap[f"{a}&{b}"] = {"shared": inter, "expected_if_random": n_top * n_top / len(index)}
            overlap[f"{a}&{b}"]["spearman_mse"] = spearman(errors[a], errors[b])
    shared_all = set(poor[args.datasets[0]].tolist())
    for d in args.datasets[1:]:
        shared_all &= set(poor[d].tolist())
    overlap["all"] = {"shared": len(shared_all)}
    results["overlap"] = overlap
    (out / "poor_recon.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "errors.npz", index=index, params=params,
             **{f"mse_{d}": errors[d] for d in args.datasets}, **{f"fh_{d}": fh[d] for d in args.datasets})

    print(f"{len(index)} val+test planets; poorly reconstructed = top {args.top_frac:.0%} ({n_top}) by own-code MSE")
    print("\nAUC = P(param of a poor planet > param of another); 0.5 = no difference. rho = Spearman(param, log MSE)")
    for d in args.datasets:
        print(f"\n{LABELS[d]} (MSE threshold {results['codes'][d]['mse_threshold']:.2e})")
        print(f"  {'':>20} {'AUC':>6} {'rho':>6} {'median poor':>12} {'median rest':>12}")
        for c in columns:
            r = results["codes"][d]["params"][c]
            print(f"  {c:>20} {r['auc_poor_vs_rest']:>6.2f} {r['spearman_logmse']:>6.2f} "
                  f"{r['median_poor']:>12.4g} {r['median_rest']:>12.4g}")
    print("\noverlap of poorly reconstructed planets:")
    for k, v in overlap.items():
        print(f"  {k}: {v}")

    plot_hists(args, columns, params, fh, poor, out / "poor_recon_params.png")
    plot_scatter(args, results, columns, params, fh, errors, out / "poor_recon_scatter.png")
    plot_t_logg(args, params, fh, poor, out / "poor_recon_T_logg.png")
    print(f"\nsaved to {out}/")


def plot_hists(args, columns, params, fh, poor, path):
    n = len(columns)
    fig, axes = plt.subplots(len(args.datasets), n, figsize=(2.3 * n, 2.4 * len(args.datasets)), squeeze=False)
    for i, d in enumerate(args.datasets):
        feats = features(params, fh[d])
        for j, c in enumerate(columns):
            ax = axes[i, j]
            v = feats[:, j]
            log = c == "feature_height_ppm"
            vv = np.log10(v) if log else v
            bins = np.linspace(vv.min(), vv.max(), 30)
            ax.hist(vv, bins=bins, density=True, color="grey", alpha=0.4, label="all")
            ax.hist(vv[poor[d]], bins=bins, density=True, histtype="step", lw=1.6, color=f"C{i}",
                    label=f"top {args.top_frac:.0%} MSE")
            ax.set_yticks([])
            if i == len(args.datasets) - 1:
                ax.set_xlabel(f"log10 {c}" if log else c, fontsize=8)
            if j == 0:
                ax.set_ylabel(LABELS[d])
            ax.tick_params(labelsize=7)
        axes[i, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Parameters of poorly reconstructed own-code spectra vs all (density)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(args, results, columns, params, fh, errors, path):
    """log MSE vs the two most influential variables per code (by |AUC - 0.5|)."""
    fig, axes = plt.subplots(len(args.datasets), 2, figsize=(10, 3.4 * len(args.datasets)), squeeze=False)
    for i, d in enumerate(args.datasets):
        res = results["codes"][d]["params"]
        top = sorted(columns, key=lambda c: -abs(res[c]["auc_poor_vs_rest"] - 0.5))[:2]
        feats = dict(zip(columns, features(params, fh[d]).T))
        for ax, c in zip(axes[i], top):
            v = feats[c]
            ax.scatter(v, np.log10(errors[d]), s=2, alpha=0.25, color=f"C{i}", rasterized=True)
            ax.axhline(np.log10(results["codes"][d]["mse_threshold"]), color="k", ls="--", lw=1)
            if c == "feature_height_ppm":
                ax.set_xscale("log")
            ax.set(xlabel=c, ylabel="log10 MSE", title=f"{LABELS[d]}: AUC={res[c]['auc_poor_vs_rest']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_t_logg(args, params, fh, poor, path):
    """Fraction of planets that are poorly reconstructed, binned in T and log g."""
    feats = features(params, fh[args.datasets[0]])
    t, logg = feats[:, 3], feats[:, 9]
    tb = np.linspace(t.min(), t.max(), 13)
    gb = np.linspace(np.percentile(logg, 0.5), np.percentile(logg, 99.5), 13)
    total, _, _ = np.histogram2d(t, logg, bins=[tb, gb])
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(5 * len(args.datasets), 4.2), squeeze=False)
    for ax, d in zip(axes[0], args.datasets):
        bad, _, _ = np.histogram2d(t[poor[d]], logg[poor[d]], bins=[tb, gb])
        frac = np.where(total >= 20, bad / np.maximum(total, 1), np.nan)
        im = ax.pcolormesh(tb, gb, frac.T, cmap="magma_r", vmin=0, vmax=0.4)
        ax.set(xlabel="T", title=LABELS[d])
    axes[0, 0].set_ylabel("log10 g  [log10 Mp/Rp^2, Jupiter units]")
    fig.colorbar(im, ax=axes[0].tolist(), label=f"fraction in worst {args.top_frac:.0%}")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
