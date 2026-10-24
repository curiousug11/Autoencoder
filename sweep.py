"""Sweep one train.py option (e.g. bottleneck size or activation) over several seeds.

Runs train.py once per (value, seed) in parallel, one thread each, then collects the
metrics into <out>/summary.json and plots test MSE per value.

Usage:
    python sweep.py --param latent --values 4 5 6 7 8 --out runs/sweep
    python sweep.py --param activation --values relu gelu tanh --seeds 42 43 44 --out runs/sweep_activation
    python sweep.py --param activation --values relu gelu --out runs/sweep_activation --summarize-only
    python sweep.py ... -- --patience 40     # extra train.py arguments after '--'
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ("best_epoch", "epochs_run", "train_time_s", "best_val_mse",
           "test_mse", "test_mse_median", "test_mse_p95")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param", default="latent", help="train.py option to sweep, without the leading '--'")
    p.add_argument("--values", nargs="+", default=[str(k) for k in range(4, 17)])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--pca-results", default="runs/pca/pca_results.json",
                   help="PCA curve overlaid when sweeping latent")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                   help="extra arguments passed to train.py after '--'")
    return p.parse_args()


def run_dir(out, param, value, seed):
    return out / f"{param}_{value}" / f"seed_{seed}"


def run_one(job, out, param, train_args):
    value, seed = job
    d = run_dir(out, param, value, seed)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "train.py", f"--{param.replace('_', '-')}", value, "--seed", str(seed),
           "--threads", "1", "--out", str(d), *train_args]
    with open(d / "train.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{param}={value} seed={seed}: {'done' if code == 0 else f'FAILED (exit {code})'}", flush=True)
    return code


def summarize(args, out):
    rows = []
    for value in args.values:
        runs = []
        for seed in args.seeds:
            path = run_dir(out, args.param, value, seed) / "metrics.json"
            if path.exists():
                m = json.loads(path.read_text())
                runs.append({k: m[k] for k in METRICS} | {"seed": seed})
        if not runs:
            continue
        test = np.array([r["test_mse"] for r in runs])
        rows.append({
            args.param: value,
            "n_seeds": len(runs),
            "test_mse_mean": float(test.mean()),
            "test_mse_std": float(test.std(ddof=1)) if len(test) > 1 else 0.0,
            "test_mse_min": float(test.min()),
            "test_mse_median_mean": float(np.mean([r["test_mse_median"] for r in runs])),
            "test_mse_p95_mean": float(np.mean([r["test_mse_p95"] for r in runs])),
            "epochs_mean": float(np.mean([r["epochs_run"] for r in runs])),
            "train_time_s_mean": float(np.mean([r["train_time_s"] for r in runs])),
            "runs": runs,
        })
    (out / "summary.json").write_text(json.dumps(rows, indent=2))

    print(f"{args.param:>12} {'seeds':>5} {'test MSE mean':>14} {'std':>9} {'min':>9} "
          f"{'median':>9} {'p95':>9} {'epochs':>7} {'time/s':>7}")
    for r in sorted(rows, key=lambda r: r["test_mse_mean"]):
        print(f"{r[args.param]:>12} {r['n_seeds']:>5} {r['test_mse_mean']:>14.2e} {r['test_mse_std']:>9.1e} "
              f"{r['test_mse_min']:>9.2e} {r['test_mse_median_mean']:>9.2e} {r['test_mse_p95_mean']:>9.2e} "
              f"{r['epochs_mean']:>7.0f} {r['train_time_s_mean']:>7.0f}")

    plot(args, rows, out / "sweep.png")


def plot(args, rows, path):
    numeric = all(v.lstrip("-").replace(".", "", 1).isdigit() for v in args.values)
    if not numeric:
        rows = sorted(rows, key=lambda r: r["test_mse_mean"])
    labels = [r[args.param] for r in rows]
    x = np.array([float(v) for v in labels]) if numeric else np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(rows) + 2), 4.2))
    for i, r in enumerate(rows):
        ys = [run["test_mse"] for run in r["runs"]]
        ax.scatter(np.full(len(ys), x[i]), ys, s=14, color="grey", alpha=0.7, zorder=2,
                   label="individual seeds" if i == 0 else None)
    ax.plot(x, [r["test_mse_mean"] for r in rows], "o" + ("-" if numeric else ""), color="C0",
            ms=8, zorder=3, label="mean test MSE")
    ax.plot(x, [r["test_mse_median_mean"] for r in rows], "s" + ("--" if numeric else ""), color="C1",
            ms=5, zorder=3, label="median test MSE")
    pca_path = Path(args.pca_results)
    if args.param == "latent" and pca_path.exists():
        pca = {str(r["k"]): r["test_mse"] for r in json.loads(pca_path.read_text())["per_k"]}
        ks = [v for v in labels if v in pca]
        ax.plot([float(k) for k in ks], [pca[k] for k in ks], "k.-", alpha=0.5, label="PCA test MSE")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=0 if numeric else 30)
    ax.set(xlabel=args.param, ylabel="reconstruction MSE (z-normalised)",
           title=f"Test reconstruction error vs {args.param} ({len(args.seeds)} seed(s))")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_args = [a for a in args.train_args if a != "--"]
    if not args.summarize_only:
        jobs = [(v, s) for s in args.seeds for v in args.values]
        with ThreadPoolExecutor(args.workers) as pool:
            codes = list(pool.map(lambda job: run_one(job, out, args.param, train_args), jobs))
        if any(codes):
            print("some runs failed; see their train.log", file=sys.stderr)
    summarize(args, out)


if __name__ == "__main__":
    main()
