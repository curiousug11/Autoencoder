"""Sweep the autoencoder bottleneck size and compare against PCA.

Runs train.py once per latent size (in parallel, one thread each), then collects
the metrics into runs/sweep/summary.json and plots test MSE vs latent size with the
PCA curve from runs/pca/pca_results.json for reference.

Usage:
    python sweep.py                         # latent 4..16, 4 parallel workers
    python sweep.py --latents 6 8 10 --workers 2
    python sweep.py --summarize-only        # re-plot from finished runs
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", type=int, nargs="+", default=list(range(4, 17)))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--pca-results", default="runs/pca/pca_results.json")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                   help="extra arguments passed to train.py after '--'")
    return p.parse_args()


def run_one(latent, out, train_args):
    run_dir = out / f"latent_{latent:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "train.py", "--latent", str(latent), "--threads", "1",
           "--out", str(run_dir), *train_args]
    with open(run_dir / "train.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"latent {latent:2d}: {'done' if code == 0 else f'FAILED (exit {code})'}", flush=True)
    return code


def summarize(latents, out, pca_path):
    rows = []
    for k in latents:
        path = out / f"latent_{k:02d}" / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        rows.append({key: m[key] for key in (
            "best_epoch", "epochs_run", "train_time_s", "best_val_mse",
            "test_mse", "test_mse_median", "test_mse_p95")} | {"latent": k})
    (out / "summary.json").write_text(json.dumps(rows, indent=2))

    print(f"{'latent':>6} {'epochs':>6} {'val MSE':>10} {'test MSE':>10} {'median':>10} {'p95':>10}")
    for r in rows:
        print(f"{r['latent']:>6} {r['epochs_run']:>6} {r['best_val_mse']:>10.2e} {r['test_mse']:>10.2e} "
              f"{r['test_mse_median']:>10.2e} {r['test_mse_p95']:>10.2e}")

    ks = [r["latent"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.semilogy(ks, [r["test_mse"] for r in rows], "o-", label="AE test MSE (mean)")
    ax.semilogy(ks, [r["test_mse_median"] for r in rows], "s--", ms=4, label="AE test MSE (median)")
    ax.semilogy(ks, [r["test_mse_p95"] for r in rows], "^:", ms=4, label="AE test MSE (p95)")
    pca_path = Path(pca_path)
    if pca_path.exists():
        pca = [r for r in json.loads(pca_path.read_text())["per_k"] if r["k"] in ks]
        ax.semilogy([r["k"] for r in pca], [r["test_mse"] for r in pca], "k.-", alpha=0.5,
                    label="PCA test MSE (mean)")
    ax.set(xlabel="bottleneck size", ylabel="reconstruction MSE (z-normalised)",
           title="Test reconstruction error", xticks=ks)
    ax.legend(fontsize=8)

    ax = axes[1]
    mse = [r["test_mse"] for r in rows]
    gain = [mse[i - 1] / mse[i] for i in range(1, len(mse))]
    ax.bar(ks[1:], gain, color="C0")
    ax.axhline(1, color="grey", lw=1)
    ax.set(xlabel="bottleneck size k", ylabel="test MSE(k-1) / test MSE(k)",
           title="Improvement from adding one latent dimension", xticks=ks[1:])

    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_args = [a for a in args.train_args if a != "--"]
    if not args.summarize_only:
        with ThreadPoolExecutor(args.workers) as pool:
            codes = list(pool.map(lambda k: run_one(k, out, train_args), args.latents))
        if any(codes):
            print("some runs failed; see their train.log", file=sys.stderr)
    summarize(args.latents, out, args.pca_results)


if __name__ == "__main__":
    main()
