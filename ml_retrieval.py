"""Machine-learning retrieval with XGBoost: self- vs cross-code retrieval.

For each training code, one XGBoost regressor per parameter (Rs, Rp, Mp, T, chem1-5) is trained
on that code's spectra. It is then applied to the held-out test planets of every code:
  - self retrieval: test spectra from the training code,
  - cross retrieval: the same planets' spectra from the other codes.
Since the planets are identical, any change between self and cross predictions is caused purely
by the differences between the codes (the code systematics).

Data: planets passing the 210 ppm cut in all codes, the shared train/val/test split (seed 42)
of the anomaly experiment. Features: the z-normalised spectrum (52 bins) plus log10 of the
spectrum's mean depth and std, so the absolute depth (which carries Rp/Rs) is not thrown away.
Optional white noise (--noise-ppm) is added to the raw spectra before computing the features.

Outputs (in --out):
  loss_curves_<code>.png      train/val RMSE vs boosting round for each parameter
  learning_curves.png         test R^2 vs training-set size, self and cross
  true_vs_pred_<code>.png     true vs predicted for every tested code, with R^2
  residual_spread.png         distribution of predicted - true for every train/test combination
  code_bias.png               median bias (predicted - true) per train/test combination
  bias_vs_true.png            bias as a function of the true value, self vs cross
  retrieval_results.json, predictions.npz

Usage:
    python ml_retrieval.py
    python ml_retrieval.py --noise-ppm 20 --out runs/retrieval_20ppm
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import xgboost as xgb

from anomaly import LABELS
from data import DATA_DIR, DATASETS, load_raw, prepare_splits, znormalise

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARAMS = ["Rs", "Rp", "Mp", "T", "chem1", "chem2", "chem3", "chem4", "chem5"]
UNITS = ["R☉", "R_J", "M_J", "K", "log VMR", "log VMR", "log VMR", "log VMR", "log VMR"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/retrieval")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--train-codes", nargs="+", default=None, choices=DATASETS, help="default: all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-ppm", type=float, default=0.0)
    p.add_argument("--n-estimators", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--train-fracs", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5],
                   help="extra training-set fractions for the learning curves (1.0 is always run)")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--replot", action="store_true",
                   help="redo the plots (except loss curves) from saved predictions/results, no training")
    return p.parse_args()


def features(raw):
    z, m, s = znormalise(raw)
    return np.column_stack([z, np.log10(m), np.log10(s)]).astype(np.float32)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def make_model(args, n_estimators=None):
    return xgb.XGBRegressor(n_estimators=n_estimators or args.n_estimators, learning_rate=args.learning_rate,
                            max_depth=args.max_depth, subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                            tree_method="hist", n_jobs=args.n_jobs, random_state=args.seed, eval_metric="rmse",
                            early_stopping_rounds=args.early_stopping)


def replot(args, out):
    results = json.loads((out / "retrieval_results.json").read_text())
    saved = np.load(out / "predictions.npz")
    train_codes = list(results["train_codes"])
    y = saved["y_true"]
    preds = {(a, b, n): saved[f"{a}_on_{b}_{n}"] for a in train_codes for b in args.datasets for n in PARAMS}
    learning = {(a, n, float(f)): v for a in train_codes for n in PARAMS
                for f, v in results["train_codes"][a][n]["learning_curve"].items()}
    for a in train_codes:
        plot_true_vs_pred(args, a, y, preds, out / f"true_vs_pred_{a}.png")
    plot_learning(args, train_codes, learning, results["n_train_planets"], out / "learning_curves.png")
    plot_spread(args, train_codes, y, preds, out / "residual_spread.png")
    plot_bias_matrix(args, train_codes, results, out / "code_bias.png")
    plot_bias_vs_true(args, train_codes, y, preds, out / "bias_vs_true.png")
    print(f"replotted in {out}/")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        return replot(args, out)
    rng = np.random.default_rng(args.seed)
    train_codes = args.train_codes or args.datasets

    splits = prepare_splits(args.data_dir, seed=args.seed, dataset=args.datasets[0], cut_datasets=args.datasets)[0]
    y = {part: splits[part].params for part in ("train", "val", "test")}
    X = {}
    for d in args.datasets:
        raw = load_raw(args.data_dir, d)[0]
        for part in ("train", "val", "test"):
            r = raw[splits[part].index]
            if args.noise_ppm:
                r = r + rng.normal(0, args.noise_ppm * 1e-6, r.shape)
            X[(d, part)] = features(r)
    prior_range = np.ptp(np.concatenate([y["train"], y["val"], y["test"]]), 0)
    n_tr = len(y["train"])
    subsets = {f: np.sort(rng.choice(n_tr, int(round(f * n_tr)), replace=False)) for f in args.train_fracs}

    preds, curves, best_iter, learning = {}, {}, {}, {}
    for a in train_codes:
        for j, name in enumerate(PARAMS):
            t0 = time.time()
            model = make_model(args)
            model.fit(X[(a, "train")], y["train"][:, j],
                      eval_set=[(X[(a, "train")], y["train"][:, j]), (X[(a, "val")], y["val"][:, j])], verbose=False)
            ev = model.evals_result()
            curves[(a, name)] = (ev["validation_0"]["rmse"], ev["validation_1"]["rmse"])
            best_iter[(a, name)] = int(model.best_iteration)
            for b in args.datasets:
                preds[(a, b, name)] = model.predict(X[(b, "test")])
            learning[(a, name, 1.0)] = {b: r2(y["test"][:, j], preds[(a, b, name)]) for b in args.datasets}
            for f, idx in subsets.items():
                m = make_model(args)
                m.fit(X[(a, "train")][idx], y["train"][idx, j], eval_set=[(X[(a, "val")], y["val"][:, j])],
                      verbose=False)
                learning[(a, name, f)] = {b: r2(y["test"][:, j], m.predict(X[(b, "test")])) for b in args.datasets}
            print(f"train {LABELS[a]:<14} {name:<6} best round {best_iter[(a, name)]:>4}  R² "
                  + "  ".join(f"{LABELS[b]} {learning[(a, name, 1.0)][b]:.3f}" for b in args.datasets)
                  + f"  [{time.time() - t0:.0f}s]", flush=True)

    results = summarise(args, train_codes, y["test"], preds, prior_range, best_iter, learning, n_tr)
    (out / "retrieval_results.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "predictions.npz", test_index=splits["test"].index, y_true=y["test"],
             **{f"{a}_on_{b}_{n}": p for (a, b, n), p in preds.items()})
    print_table(args, train_codes, results)

    for a in train_codes:
        plot_loss(a, curves, best_iter, out / f"loss_curves_{a}.png")
        plot_true_vs_pred(args, a, y["test"], preds, out / f"true_vs_pred_{a}.png")
    plot_learning(args, train_codes, learning, n_tr, out / "learning_curves.png")
    plot_spread(args, train_codes, y["test"], preds, out / "residual_spread.png")
    plot_bias_matrix(args, train_codes, results, out / "code_bias.png")
    plot_bias_vs_true(args, train_codes, y["test"], preds, out / "bias_vs_true.png")
    print(f"saved to {out}/")


def summarise(args, train_codes, y, preds, prior_range, best_iter, learning, n_tr):
    res = {"noise_ppm": args.noise_ppm, "n_test_planets": int(len(y)), "n_train_planets": int(n_tr),
           "params": PARAMS, "prior_range": dict(zip(PARAMS, prior_range.tolist())), "train_codes": {}}
    for a in train_codes:
        ra = {}
        for j, name in enumerate(PARAMS):
            self_pred = preds[(a, a, name)]
            entry = {"best_round": best_iter[(a, name)], "tested_on": {},
                     "learning_curve": {f"{f:g}": v for (aa, n, f), v in learning.items() if aa == a and n == name}}
            for b in args.datasets:
                p = preds[(a, b, name)]
                err = p - y[:, j]
                shift = p - self_pred  # change caused only by the code, same planets
                entry["tested_on"][b] = {
                    "r2": r2(y[:, j], p), "rmse": float(np.sqrt((err**2).mean())),
                    "median_bias": float(np.median(err)), "mean_bias": float(err.mean()),
                    "spread_16_84": float((np.percentile(err, 84) - np.percentile(err, 16)) / 2),
                    "median_bias_pct_range": float(100 * np.median(err) / prior_range[j]),
                    "median_shift_vs_self": float(np.median(shift)),
                    "abs_shift_vs_self_median": float(np.median(np.abs(shift))),
                }
            ra[name] = entry
        res["train_codes"][a] = ra
    return res


def print_table(args, train_codes, results):
    for a in train_codes:
        print(f"\ntrained on {LABELS[a]}: R² / median bias (predicted − true) on test planets")
        print(f"  {'param':<7}" + "".join(f"{LABELS[b]:>26}" for b in args.datasets))
        for name in PARAMS:
            row = f"  {name:<7}"
            for b in args.datasets:
                t = results["train_codes"][a][name]["tested_on"][b]
                row += f"{t['r2']:>12.3f} {t['median_bias']:>+12.3g} "
            print(row)


def plot_loss(a, curves, best_iter, path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, name in zip(axes.flat, PARAMS):
        tr, va = curves[(a, name)]
        ax.plot(tr, label="train")
        ax.plot(va, label="val")
        ax.axvline(best_iter[(a, name)], color="grey", ls="--", lw=1, label="best")
        ax.set(yscale="log", title=name, xlabel="boosting round", ylabel="RMSE")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"XGBoost loss curves, trained on {LABELS[a]}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning(args, train_codes, learning, n_tr, path):
    fracs = sorted({f for (_, _, f) in learning})
    fig, axes = plt.subplots(len(train_codes), len(PARAMS), figsize=(2.6 * len(PARAMS), 2.6 * len(train_codes)),
                             squeeze=False, sharex=True)
    for i, a in enumerate(train_codes):
        for j, name in enumerate(PARAMS):
            ax = axes[i, j]
            for k, b in enumerate(args.datasets):
                vals = [learning[(a, name, f)][b] for f in fracs]
                ax.plot(np.array(fracs) * n_tr, vals, "o-" if b == a else "s--", ms=3, color=f"C{k}",
                        lw=2 if b == a else 1, label=f"on {LABELS[b]}" + (" (self)" if b == a else ""))
            ax.set_xscale("log")
            ax.set_xticks([1e3, 1e4], ["1k", "10k"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f"trained on {LABELS[a]}\nR²", fontsize=8)
            if i == len(train_codes) - 1:
                ax.set_xlabel("training planets", fontsize=8)
        axes[i, 0].legend(fontsize=6)
    fig.suptitle("Learning curves: test R² vs training-set size (solid = self, dashed = cross)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_true_vs_pred(args, a, y, preds, path):
    fig, axes = plt.subplots(len(args.datasets), len(PARAMS), figsize=(2.5 * len(PARAMS), 2.6 * len(args.datasets)),
                             squeeze=False)
    for i, b in enumerate(args.datasets):
        for j, name in enumerate(PARAMS):
            ax = axes[i, j]
            t, p = y[:, j], preds[(a, b, name)]
            lo, hi = t.min(), t.max()
            ax.hexbin(t, p, gridsize=35, extent=(lo, hi, lo, hi), bins="log", cmap="viridis", mincnt=1)
            ax.plot([lo, hi], [lo, hi], "r-", lw=0.8)
            ax.text(0.04, 0.93, f"R²={r2(t, p):.3f}", transform=ax.transAxes, fontsize=7, color="w",
                    bbox=dict(facecolor="k", alpha=0.5, lw=0))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f"tested on {LABELS[b]}" + (" (self)" if b == a else "") + "\npredicted", fontsize=8)
            if i == len(args.datasets) - 1:
                ax.set_xlabel("true", fontsize=8)
    fig.suptitle(f"True vs predicted, XGBoost trained on {LABELS[a]}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_spread(args, train_codes, y, preds, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, (j, name) in zip(axes.flat, enumerate(PARAMS)):
        data, pos, colors, ticks = [], [], [], []
        for i, a in enumerate(train_codes):
            for k, b in enumerate(args.datasets):
                data.append(preds[(a, b, name)] - y[:, j])
                pos.append(i * (len(args.datasets) + 1) + k)
                colors.append(f"C{k}")
            ticks.append((i * (len(args.datasets) + 1) + (len(args.datasets) - 1) / 2, LABELS[a]))
        bp = ax.boxplot(data, positions=pos, widths=0.7, showfliers=False, patch_artist=True,
                        whis=(5, 95), medianprops=dict(color="k"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks([t for t, _ in ticks], [f"train {lbl}" for _, lbl in ticks], fontsize=8)
        ax.set(title=name, ylabel=f"predicted − true ({UNITS[j]})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=f"C{k}", alpha=0.6) for k in range(len(args.datasets))]
    fig.suptitle("Spread of predicted − true (box 25–75%, whiskers 5–95%)")
    fig.legend(handles, [f"tested on {LABELS[b]}" for b in args.datasets], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bias_matrix(args, train_codes, results, path):
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.8))
    lab = [LABELS[d] for d in args.datasets]
    for ax, name in zip(axes.flat, PARAMS):
        m = np.array([[results["train_codes"][a][name]["tested_on"][b]["median_bias_pct_range"]
                       for b in args.datasets] for a in train_codes])
        lim = max(1.0, np.abs(m).max())
        ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                ax.text(c, r, f"{m[r, c]:+.1f}%", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(lab)), lab, rotation=30, fontsize=7)
        ax.set_yticks(range(len(train_codes)), [LABELS[a] for a in train_codes], fontsize=7)
        ax.set(title=name, xlabel="tested on")
    axes.flat[0].set_ylabel("trained on")
    axes.flat[-1].axis("off")
    fig.suptitle("Median bias (predicted − true) as % of the parameter's prior range; diagonal = self retrieval")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bias_vs_true(args, train_codes, y, preds, path, n_bins=12):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    styles = ["-", "--", ":"]
    for ax, (j, name) in zip(axes.flat, enumerate(PARAMS)):
        t = y[:, j]
        edges = np.quantile(t, np.linspace(0, 1, n_bins + 1))
        which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
        centres = np.array([np.median(t[which == i]) for i in range(n_bins)])
        for i, a in enumerate(train_codes):
            for k, b in enumerate(args.datasets):
                err = preds[(a, b, name)] - t
                med = [np.median(err[which == q]) for q in range(n_bins)]
                ax.plot(centres, med, styles[i % 3], color=f"C{k}", lw=2.2 if a == b else 1.2,
                        label=f"train {LABELS[a]} → {LABELS[b]}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(title=name, xlabel=f"true {name} ({UNITS[j]})", ylabel="median predicted − true")
    axes.flat[0].legend(fontsize=6, ncol=1)
    fig.suptitle("Bias vs true value (line style = training code, colour = tested code, thick = self)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
