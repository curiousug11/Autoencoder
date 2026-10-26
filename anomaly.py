"""Cross-code anomaly detection with autoencoders.

For each code (POS, petitRADTRANS, TauREx), train an autoencoder on that code's spectra only.
Then score the held-out test planets from every code by per-spectrum reconstruction MSE:
spectra from the training code are "normal", spectra from the other two are "anomalous".

All codes use the same planets: a planet is kept only if it passes the 210 ppm feature-height
cut in all three codes, each spectrum is z-normalised, and all codes share one
train/val/test split. The test planets were never seen in training, in any code.

Usage:
    python anomaly.py                    # train the 3 models (in parallel) and evaluate
    python anomaly.py --eval-only        # re-evaluate existing models
    python anomaly.py --train-datasets pos prt --init-seeds 43 44   # extra seeds for some codes
    python anomaly.py -- --epochs 50     # extra train.py arguments after '--'
    python anomaly.py --out runs/anomaly_noise/sigma20 --no-eval -- --noise-ppm 20   # denoising models

When a code has several trained runs (different init seeds), evaluation uses the one with
the lowest validation loss on that code's own spectra; no anomalous data is used to choose.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch

from data import DATA_DIR, DATASETS, prepare_splits
from model import ACTIVATIONS, Autoencoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"pos": "POS", "prt": "petitRADTRANS", "taurex": "TauREx"}
TRAIN_ARGS = ["--hidden", "256", "128", "64", "32", "16", "--latent", "8",
              "--scheduler", "cosine", "--patience", "50"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/anomaly")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--seed", type=int, default=42, help="split seed, shared by all codes")
    p.add_argument("--train-datasets", nargs="+", default=None, choices=DATASETS,
                   help="codes to train (default: all of --datasets)")
    p.add_argument("--init-seeds", type=int, nargs="+", default=None,
                   help="weight-init seeds to train for each code (default: --seed)")
    p.add_argument("--min-feature-height-ppm", type=float, default=210.0)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="only train (e.g. denoising models, see chi2_score.py)")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                   help="extra arguments passed to train.py after '--'")
    return p.parse_args()


def run_dir(out, dataset, init_seed, split_seed):
    name = f"train_{dataset}" if init_seed == split_seed else f"train_{dataset}_seed{init_seed}"
    return Path(out) / name


def train_one(job, args, extra):
    dataset, init_seed = job
    d = run_dir(args.out, dataset, init_seed, args.seed)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "train.py", "--dataset", dataset, "--cut-datasets", *args.datasets,
           "--seed", str(init_seed), "--split-seed", str(args.seed),
           "--min-feature-height-ppm", str(args.min_feature_height_ppm),
           "--data-dir", args.data_dir, "--threads", "1", "--out", str(d), *TRAIN_ARGS, *extra]
    with open(d / "train.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"train {dataset} seed {init_seed}: {'done' if code == 0 else f'FAILED (exit {code})'}", flush=True)
    return code


def select_run(out, dataset):
    """Among the finished runs for `dataset`, return the one with the lowest own-code val loss."""
    runs = [p.parent for p in Path(out).glob(f"train_{dataset}*/metrics.json")
            if p.parent.name == f"train_{dataset}" or p.parent.name.startswith(f"train_{dataset}_seed")]
    vals = {r: json.loads((r / "metrics.json").read_text())["best_val_mse"] for r in runs}
    return min(vals, key=vals.get), {r.name: v for r, v in sorted(vals.items())}


def load_model(path):
    ckpt = torch.load(path)
    a = ckpt["args"]
    model = Autoencoder(ckpt["n_in"], tuple(a["hidden"]), a["latent"], ACTIVATIONS[a.get("activation", "gelu")])
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


@torch.no_grad()
def per_spectrum_mse(model, x):
    return ((model(torch.from_numpy(x)).numpy() - x) ** 2).mean(axis=1)


def roc_auc(normal, anomalous):
    """P(score of a random anomalous spectrum > score of a random normal one)."""
    scores = np.concatenate([normal, anomalous])
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores)] = np.arange(1, len(scores) + 1)
    r_anom = ranks[len(normal):].sum()
    n0, n1 = len(normal), len(anomalous)
    return float((r_anom - n1 * (n1 + 1) / 2) / (n0 * n1))


def roc_curve(normal, anomalous):
    thresholds = np.sort(np.concatenate([normal, anomalous]))[::-1]
    fpr = np.searchsorted(np.sort(normal), thresholds, side="left")
    tpr = np.searchsorted(np.sort(anomalous), thresholds, side="left")
    return 1 - fpr / len(normal), 1 - tpr / len(anomalous)


def evaluate(args):
    out = Path(args.out)
    data = {d: prepare_splits(args.data_dir, args.min_feature_height_ppm, seed=args.seed,
                              dataset=d, cut_datasets=args.datasets)[0] for d in args.datasets}
    test_index = data[args.datasets[0]]["test"].index
    assert all(np.array_equal(s["test"].index, test_index) for s in data.values())

    scores, results = {}, {}
    for trained in args.datasets:
        chosen, candidates = select_run(out, trained)
        model = load_model(chosen / "model.pt")
        val_err = per_spectrum_mse(model, data[trained]["val"].spectra)
        thresholds = {q: float(np.percentile(val_err, q)) for q in (95, 99)}
        scores[trained] = {d: per_spectrum_mse(model, data[d]["test"].spectra) for d in args.datasets}
        normal = scores[trained][trained]
        res = {"model_run": chosen.name, "candidate_val_mse": candidates,
               "thresholds_from_val": {f"p{q}": t for q, t in thresholds.items()}, "tested_on": {}}
        for d in args.datasets:
            s = scores[trained][d]
            r = {"mse_mean": float(s.mean()), "mse_median": float(np.median(s)),
                 **{f"flagged_frac_p{q}": float((s > t).mean()) for q, t in thresholds.items()}}
            if d != trained:
                r["auc"] = roc_auc(normal, s)
            res["tested_on"][d] = r
        results[trained] = res
    results["n_test_planets"] = int(len(test_index))
    (out / "anomaly_results.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "test_scores.npz", test_index=test_index,
             **{f"{a}_on_{b}": scores[a][b] for a in args.datasets for b in args.datasets})

    print(f"test planets: {len(test_index)}")
    for trained in args.datasets:
        r = results[trained]
        print(f"\ntrained on {LABELS[trained]}, model {r['model_run']} (val p95 threshold {r['thresholds_from_val']['p95']:.2e}, "
              f"p99 {r['thresholds_from_val']['p99']:.2e})")
        print(f"  {'tested on':<15} {'median MSE':>11} {'mean MSE':>10} {'AUC':>6} {'>p95':>7} {'>p99':>7}")
        for d, t in r["tested_on"].items():
            auc = f"{t['auc']:.3f}" if "auc" in t else "  -  "
            print(f"  {LABELS[d]:<15} {t['mse_median']:>11.2e} {t['mse_mean']:>10.2e} {auc:>6} "
                  f"{t['flagged_frac_p95']:>7.1%} {t['flagged_frac_p99']:>7.1%}")

    plot_histograms(args.datasets, scores, results, out / "anomaly_histograms.png")
    plot_roc(args.datasets, scores, results, out / "anomaly_roc.png")
    print(f"\nsaved to {out}/")


def plot_histograms(datasets, scores, results, path):
    all_scores = np.concatenate([s for row in scores.values() for s in row.values()])
    bins = np.linspace(np.log10(all_scores.min()), np.log10(all_scores.max()), 80)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), sharey=True)
    for ax, trained in zip(np.atleast_1d(axes), datasets):
        for i, d in enumerate(datasets):
            normal = d == trained
            auc = results[trained]["tested_on"][d].get("auc")
            label = f"{LABELS[d]} ({'normal' if normal else f'anomalous, AUC={auc:.3f}'})"
            ax.hist(np.log10(scores[trained][d]), bins=bins, histtype="stepfilled" if normal else "step",
                    alpha=0.35 if normal else 1, lw=1.8, color=f"C{i}", label=label)
        t = results[trained]["thresholds_from_val"]["p99"]
        ax.axvline(np.log10(t), color="k", ls="--", lw=1, label="99th pct of normal val error")
        ax.set(title=f"AE trained on {LABELS[trained]}", xlabel="log10 reconstruction MSE (z-normalised)")
        ax.legend(fontsize=8, loc="upper right")
    np.atleast_1d(axes)[0].set_ylabel("test spectra")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(datasets, scores, results, path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.2), sharey=True)
    for ax, trained in zip(np.atleast_1d(axes), datasets):
        for i, d in enumerate(datasets):
            if d == trained:
                continue
            fpr, tpr = roc_curve(scores[trained][trained], scores[trained][d])
            ax.plot(fpr, tpr, color=f"C{i}", label=f"{LABELS[d]} (AUC={results[trained]['tested_on'][d]['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set(title=f"AE trained on {LABELS[trained]}", xlabel="false positive rate (normal flagged)")
        ax.legend(fontsize=8, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("true positive rate (anomalous flagged)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    extra = [a for a in args.train_args if a != "--"]
    if not args.eval_only:
        jobs = [(d, s) for d in (args.train_datasets or args.datasets) for s in (args.init_seeds or [args.seed])]
        with ThreadPoolExecutor(min(len(jobs), 4)) as pool:
            codes = list(pool.map(lambda job: train_one(job, args, extra), jobs))
        if any(codes):
            sys.exit("training failed; see train.log in the run directories")
    if not args.no_eval:
        evaluate(args)


if __name__ == "__main__":
    main()
