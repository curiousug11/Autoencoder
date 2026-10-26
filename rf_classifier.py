"""Random-forest classifier: which code (POS, petitRADTRANS, TauREx) produced a spectrum?

Same planets and split as the anomaly experiment (anomaly.py): planets passing the 210 ppm
feature-height cut in all three codes, shared train/val/test split (seed 42). Every planet
contributes one spectrum per code, labelled by code, so the classifier must learn the
differences between codes, and test planets are never seen in training (in any code).

For each noise level sigma (ppm), white Gaussian noise is added to the raw spectra (train and
test alike, a fresh draw for each copy), then each spectrum is z-normalised as before. The
forest is trained on the noisy train set and evaluated on the noisy test set.

Pairwise AUCs (score p_b / (p_a + p_b) on test spectra of codes a and b) are directly
comparable to the anomaly-detection AUCs in chi2_score.py.

Usage:
    python rf_classifier.py
    python rf_classifier.py --sigmas 0 20 --n-estimators 500 --train-copies 2
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from anomaly import LABELS, roc_auc
from data import DATA_DIR, DATASETS, load_raw, prepare_splits, znormalise

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/rf_classifier")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--seed", type=int, default=42, help="split seed (same as anomaly.py)")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0, 10, 20, 30, 50], help="noise per bin, ppm")
    p.add_argument("--train-copies", type=int, default=1,
                   help="noisy copies of each training spectrum (independent noise draws)")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--max-features", default="sqrt")
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def make_xy(raw, sigma, rng, copies=1):
    """Stack all codes, add noise, z-normalise. raw: {code: (n, bins)} in transit depth."""
    xs, ys = [], []
    for label, spectra in enumerate(raw.values()):
        for _ in range(copies):
            noisy = spectra + rng.normal(0, sigma * 1e-6, spectra.shape) if sigma else spectra
            xs.append(znormalise(noisy)[0].astype(np.float32))
            ys.append(np.full(len(spectra), label))
    return np.concatenate(xs), np.concatenate(ys)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    splits, wavelength, info = prepare_splits(args.data_dir, seed=args.seed, dataset=args.datasets[0],
                                              cut_datasets=args.datasets)
    full = {d: load_raw(args.data_dir, d)[0] for d in args.datasets}
    raw = {part: {d: full[d][splits[part].index] for d in args.datasets} for part in ("train", "test")}
    names = [LABELS[d] for d in args.datasets]
    k = len(args.datasets)

    results = {"n_train_planets": int(len(splits["train"].index)), "n_test_planets": int(len(splits["test"].index)),
               "args": vars(args), "per_sigma": {}}
    importances, confusions = {}, {}
    for sigma in args.sigmas:
        t0 = time.time()
        x_tr, y_tr = make_xy(raw["train"], sigma, rng, args.train_copies)
        x_te, y_te = make_xy(raw["test"], sigma, rng)
        max_features = args.max_features if args.max_features in ("sqrt", "log2") else float(args.max_features)
        rf = RandomForestClassifier(n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf,
                                    max_features=max_features, n_jobs=args.n_jobs, random_state=args.seed)
        rf.fit(x_tr, y_tr)
        proba = rf.predict_proba(x_te)
        pred = proba.argmax(1)
        cm = confusion_matrix(y_te, pred, labels=range(k))
        pair_auc = {}
        for a in range(k):
            for b in range(k):
                if a != b:
                    sel_a, sel_b = y_te == a, y_te == b
                    score = lambda s: proba[s, b] / np.clip(proba[s, a] + proba[s, b], 1e-12, None)  # noqa: E731
                    pair_auc[f"{args.datasets[a]}->{args.datasets[b]}"] = roc_auc(score(sel_a), score(sel_b))
        # Per-planet: are all three codes of a test planet classified correctly?
        n_te = len(splits["test"].index)
        all_right = (pred.reshape(k, n_te) == np.arange(k)[:, None]).all(0)
        r = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "train_accuracy": float(rf.score(x_tr[::7], y_tr[::7])),
            "macro_auc_ovr": float(roc_auc_score(y_te, proba, multi_class="ovr")),
            "per_class_recall": {args.datasets[i]: float(cm[i, i] / cm[i].sum()) for i in range(k)},
            "pair_auc": pair_auc,
            "mean_pair_auc": float(np.mean(list(pair_auc.values()))),
            "planets_all_codes_correct": float(all_right.mean()),
            "confusion": cm.tolist(),
            "fit_time_s": round(time.time() - t0, 1),
        }
        results["per_sigma"][f"{sigma:g}"] = r
        importances[sigma] = rf.feature_importances_
        confusions[sigma] = cm
        np.save(out / f"test_proba_sigma{sigma:g}.npy", proba.astype(np.float32))
        print(f"sigma {sigma:>3g} ppm: accuracy {r['accuracy']:.3f} (train {r['train_accuracy']:.3f}), "
              f"macro AUC {r['macro_auc_ovr']:.3f}, mean pair AUC {r['mean_pair_auc']:.3f}, "
              f"recall " + " ".join(f"{LABELS[d]} {v:.2f}" for d, v in r["per_class_recall"].items())
              + f"  [{r['fit_time_s']:.0f}s]", flush=True)

    (out / "rf_results.json").write_text(json.dumps(results, indent=2))
    np.save(out / "test_index.npy", splits["test"].index)
    plot_summary(args, results, out / "rf_accuracy_vs_noise.png")
    plot_confusions(args, names, confusions, out / "rf_confusion.png")
    plot_importances(args, wavelength, importances, out / "rf_importance.png")
    print(f"saved to {out}/")


def plot_summary(args, results, path):
    x = np.array(args.sigmas, float)
    xplot = np.where(x == 0, 5, x)
    per = [results["per_sigma"][f"{s:g}"] for s in args.sigmas]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(xplot, [r["accuracy"] for r in per], "ko-", label="overall")
    for i, d in enumerate(args.datasets):
        ax.plot(xplot, [r["per_class_recall"][d] for r in per], "o--", ms=4, color=f"C{i}", label=f"{LABELS[d]} recall")
    ax.axhline(1 / len(args.datasets), color="grey", lw=0.8, label="chance")
    ax.set(xscale="log", ylim=(0, 1.02), xlabel="noise per bin σ (ppm)", ylabel="accuracy", title="3-class accuracy")
    ax.set_xticks(xplot, [f"{s:g}" for s in x])
    ax.legend(fontsize=8)
    ax = axes[1]
    for p in per[0]["pair_auc"]:
        ax.plot(xplot, [r["pair_auc"][p] for r in per], "o-", ms=3, lw=1, label=p.replace("->", " vs "))
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set(xscale="log", ylim=(0.45, 1.02), xlabel="noise per bin σ (ppm)", ylabel="AUC",
           title="Pairwise AUC (comparable to anomaly-detection AUC)")
    ax.set_xticks(xplot, [f"{s:g}" for s in x])
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusions(args, names, confusions, path):
    fig, axes = plt.subplots(1, len(confusions), figsize=(3.6 * len(confusions), 3.6))
    for ax, (sigma, cm) in zip(np.atleast_1d(axes), confusions.items()):
        frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center",
                        color="white" if frac[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(len(names)), names, rotation=30, fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set(title=f"σ = {sigma:g} ppm", xlabel="predicted")
    np.atleast_1d(axes)[0].set_ylabel("true code")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importances(args, wavelength, importances, path):
    order = np.argsort(wavelength)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (sigma, imp) in enumerate(importances.items()):
        ax.plot(wavelength[order], imp[order], "o-", ms=3, lw=1, color=plt.cm.viridis(i / max(1, len(importances) - 1)),
                label=f"σ = {sigma:g} ppm")
    ax.set(xscale="log", xlabel="wavelength (µm)", ylabel="feature importance (impurity)",
           title="Which wavelength bins separate the codes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
