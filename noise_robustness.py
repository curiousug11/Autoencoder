"""How does cross-code anomaly detection degrade with observational noise?

Takes the trained anomaly-experiment models (see anomaly.py) and the same held-out test
planets, adds white Gaussian noise of sigma ppm to every wavelength bin of the raw spectra,
z-normalises, and scores with reconstruction MSE as before. Normal = noisy spectra from the
training code, anomalous = noisy spectra from the other codes. The models are not retrained.

It also reports the intrinsic size of the code differences: the RMS difference (ppm) between
two codes' noiseless spectra of the same planet. When the noise is much larger than this, no
single-spectrum method can tell the codes apart.

Usage:
    python noise_robustness.py
    python noise_robustness.py --sigmas 0 5 10 20 50 --repeats 3
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from anomaly import LABELS, load_model, per_spectrum_mse, roc_auc, select_run
from data import DATA_DIR, DATASETS, load_raw, prepare_splits, znormalise

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--runs", default="runs/anomaly")
    p.add_argument("--out", default="runs/anomaly/noise")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--seed", type=int, default=42, help="split seed used by anomaly.py")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0, 2, 5, 10, 20, 30, 50, 100],
                   help="white noise per bin, ppm")
    p.add_argument("--repeats", type=int, default=2, help="noise realisations per sigma")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    test_index = prepare_splits(args.data_dir, seed=args.seed, dataset=args.datasets[0],
                                cut_datasets=args.datasets)[0]["test"].index
    raw = {d: load_raw(args.data_dir, d)[0][test_index] for d in args.datasets}
    models = {d: load_model(select_run(args.runs, d)[0] / "model.pt") for d in args.datasets}

    # Intrinsic code differences in ppm, after removing each spectrum's mean (z-norm removes
    # the offset anyway, so a constant offset between codes is not detectable).
    diffs = {}
    for i, a in enumerate(args.datasets):
        for b in args.datasets[i + 1:]:
            da = raw[a] - raw[a].mean(1, keepdims=True)
            db = raw[b] - raw[b].mean(1, keepdims=True)
            rms = np.sqrt(((da - db) ** 2).mean(1)) * 1e6
            diffs[f"{a}-{b}"] = {"median_ppm": float(np.median(rms)), "p10_ppm": float(np.percentile(rms, 10)),
                                 "p90_ppm": float(np.percentile(rms, 90))}

    aucs = {a: {b: [] for b in args.datasets if b != a} for a in args.datasets}
    for sigma in args.sigmas:
        acc = {a: {b: [] for b in args.datasets if b != a} for a in args.datasets}
        for _ in range(args.repeats if sigma > 0 else 1):
            noisy = {d: znormalise(raw[d] + rng.normal(0, sigma * 1e-6, raw[d].shape))[0].astype(np.float32)
                     for d in args.datasets}
            for a in args.datasets:
                scores = {d: per_spectrum_mse(models[a], noisy[d]) for d in args.datasets}
                for b in acc[a]:
                    acc[a][b].append(roc_auc(scores[a], scores[b]))
        for a in args.datasets:
            for b in acc[a]:
                aucs[a][b].append(float(np.mean(acc[a][b])))
        print(f"sigma {sigma:>5g} ppm: " + "  ".join(
            f"{a}->{b} {aucs[a][b][-1]:.3f}" for a in args.datasets for b in aucs[a]), flush=True)

    results = {"sigmas_ppm": args.sigmas, "auc": aucs, "code_difference_rms_ppm": diffs,
               "n_test_planets": int(len(test_index))}
    (out / "noise_results.json").write_text(json.dumps(results, indent=2))
    print("\nRMS difference between codes' mean-subtracted noiseless spectra (ppm):")
    for k, v in diffs.items():
        print(f"  {k}: median {v['median_ppm']:.0f}  (10-90%: {v['p10_ppm']:.0f}-{v['p90_ppm']:.0f})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.array(args.sigmas, float)
    xplot = np.where(x == 0, x[x > 0].min() / 2 if (x > 0).any() else 1, x)
    for i, a in enumerate(args.datasets):
        for j, b in enumerate(aucs[a]):
            ax.plot(xplot, aucs[a][b], "o-" if j == 0 else "s--", color=f"C{args.datasets.index(a)}", ms=4,
                    label=f"trained {LABELS[a]} vs {LABELS[b]}")
    for k, v in diffs.items():
        ax.axvline(v["median_ppm"], color="grey", ls=":", lw=1)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xplot, ["0"] + [f"{s:g}" for s in x[1:]] if x[0] == 0 else [f"{s:g}" for s in x])
    ax.set(xlabel="white noise per bin σ (ppm)", ylabel="AUC (normal vs other code)", ylim=(0.45, 1.0),
           title="Cross-code anomaly detection vs noise\n(dotted: median RMS difference between codes)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "auc_vs_noise.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
