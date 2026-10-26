"""Random-forest code-classification accuracy broken down by planet type.

Uses the test-set class probabilities saved by rf_classifier.py (same test planets, one
spectrum per code). A planet's accuracy is the fraction of its three spectra (POS,
petitRADTRANS, TauREx) assigned to the right code. Accuracy is then binned by planet
parameters (Rs, Rp, Mp, T, log abundances), derived quantities (log g, feature height) and
the intrinsic size of the code differences for that planet.

Usage:
    python rf_by_planet.py
    python rf_by_planet.py --rf-dir runs/rf_classifier --sigmas 0 20 50
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from data import DATA_DIR, DATASETS, feature_height, load_raw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARAM_NAMES = ["Rs", "Rp", "Mp", "T", "chem1", "chem2", "chem3", "chem4", "chem5"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--rf-dir", default="runs/rf_classifier")
    p.add_argument("--out", default="runs/rf_classifier/by_planet")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--sigmas", type=float, nargs="+", default=None, help="default: all saved by rf_classifier.py")
    p.add_argument("--n-bins", type=int, default=8, help="quantile bins per quantity")
    return p.parse_args()


def planet_quantities(data_dir, datasets, index):
    params = load_raw(data_dir, datasets[0])[1][index]
    raw = {d: load_raw(data_dir, d)[0][index] for d in datasets}
    q = {name: params[:, i] for i, name in enumerate(PARAM_NAMES)}
    q["log_g"] = np.log10(params[:, 2] / params[:, 1] ** 2)
    q["feature_height_ppm"] = np.mean([feature_height(raw[d]) for d in datasets], 0) * 1e6
    # RMS difference between codes' mean-subtracted spectra, averaged over code pairs (ppm).
    centred = {d: raw[d] - raw[d].mean(1, keepdims=True) for d in datasets}
    pairs = [(a, b) for i, a in enumerate(datasets) for b in datasets[i + 1:]]
    q["code_diff_rms_ppm"] = np.mean([np.sqrt(((centred[a] - centred[b]) ** 2).mean(1)) for a, b in pairs], 0) * 1e6
    return q


def binned(x, acc, n_bins, log=False):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centres = [np.median(x[which == i]) for i in range(len(edges) - 1)]
    means = [acc[which == i].mean() for i in range(len(edges) - 1)]
    return np.array(centres), np.array(means), edges


def main():
    args = parse_args()
    rf_dir, out = Path(args.rf_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = np.load(rf_dir / "test_index.npy")
    sigmas = args.sigmas or json.loads((rf_dir / "rf_results.json").read_text())["args"]["sigmas"]
    k, n = len(args.datasets), len(index)
    q = planet_quantities(args.data_dir, args.datasets, index)

    acc = {}
    for s in sigmas:
        proba = np.load(rf_dir / f"test_proba_sigma{s:g}.npy")
        pred = proba.argmax(1).reshape(k, n)
        acc[s] = (pred == np.arange(k)[:, None]).mean(0)  # per planet, fraction of its 3 spectra correct

    log_x = {"feature_height_ppm", "code_diff_rms_ppm"}
    summary = {"n_test_planets": int(n), "sigmas": sigmas, "by_quantity": {}}
    print(f"{n} test planets. Accuracy (fraction of a planet's 3 spectra correctly attributed), "
          f"lowest vs highest {args.n_bins}-quantile bin:")
    print(f"  {'quantity':<20}" + "".join(f"{f'σ={s:g}':>16}" for s in sigmas))
    spread = {}
    for name, x in q.items():
        row, entry = f"  {name:<20}", {}
        for s in sigmas:
            c, m, edges = binned(x, acc[s], args.n_bins)
            entry[f"{s:g}"] = {"bin_edges": edges.tolist(), "bin_median": c.tolist(), "accuracy": m.tolist()}
            row += f"{m[0]:>8.2f} → {m[-1]:<5.2f}"
        summary["by_quantity"][name] = entry
        spread[name] = max(abs(entry[f"{s:g}"]["accuracy"][-1] - entry[f"{s:g}"]["accuracy"][0]) for s in sigmas)
        print(row)

    # Accuracy for the planets with the largest intrinsic code differences, per noise level.
    thresholds = [10, 30, 50, 100]
    summary["by_code_diff_threshold"] = {
        f"{t}": {"frac_planets": float((q["code_diff_rms_ppm"] >= t).mean()),
                 **{f"{s:g}": float(acc[s][q["code_diff_rms_ppm"] >= t].mean()) for s in sigmas}}
        for t in thresholds}
    print("\naccuracy for planets whose codes differ by at least X ppm RMS:")
    print(f"  {'X':>5} {'planets':>8}" + "".join(f"{f'σ={s:g}':>8}" for s in sigmas))
    for t, v in summary["by_code_diff_threshold"].items():
        print(f"  {t:>5} {v['frac_planets']:>8.1%}" + "".join(f"{v[f'{s:g}']:>8.2f}" for s in sigmas))
    (out / "rf_by_planet.json").write_text(json.dumps(summary, indent=2))

    plot_1d(args, q, acc, sigmas, log_x, sorted(spread, key=spread.get, reverse=True), out / "rf_acc_by_quantity.png")
    plot_2d(args, q, acc, sigmas, out / "rf_acc_T_logg.png")
    print(f"\nsaved to {out}/")


def plot_1d(args, q, acc, sigmas, log_x, order, path):
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigmas)))
    for ax, name in zip(axes.flat, order):
        for s, c in zip(sigmas, colors):
            x, m, _ = binned(q[name], acc[s], args.n_bins)
            ax.plot(x, m, "o-", ms=3, color=c, label=f"σ={s:g} ppm")
        ax.axhline(1 / len(args.datasets), color="grey", lw=0.8)
        if name in log_x:
            ax.set_xscale("log")
        ax.set(ylim=(0.2, 1.02), xlabel=name, title=name, ylabel="accuracy")
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    fig.suptitle("RF code-classification accuracy vs planet quantity (quantile bins; grey = chance)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_2d(args, q, acc, sigmas, path):
    t, g = q["T"], q["log_g"]
    tb = np.linspace(t.min(), t.max(), 11)
    gb = np.linspace(np.percentile(g, 0.5), np.percentile(g, 99.5), 11)
    count, _, _ = np.histogram2d(t, g, bins=[tb, gb])
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.3 * len(sigmas), 3.9), squeeze=False)
    for ax, s in zip(axes[0], sigmas):
        tot, _, _ = np.histogram2d(t, g, bins=[tb, gb], weights=acc[s])
        frac = np.where(count >= 15, tot / np.maximum(count, 1), np.nan)
        im = ax.pcolormesh(tb, gb, frac.T, cmap="viridis", vmin=1 / len(args.datasets), vmax=1)
        ax.set(title=f"σ = {s:g} ppm", xlabel="T")
    axes[0, 0].set_ylabel("log10 g  [Mp/Rp², Jupiter units]")
    fig.colorbar(im, ax=axes[0].tolist(), label="accuracy")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
