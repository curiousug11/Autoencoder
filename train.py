"""Train the spectrum autoencoder on the POS binned spectra.

Usage:
    python train.py                       # defaults below
    python train.py --lr 5e-4 --out runs/lr5e-4
"""

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

from data import DATA_DIR, make_loader, prepare_splits
from model import Autoencoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/ae_baseline")
    p.add_argument("--min-feature-height-ppm", type=float, default=210.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32, 16])
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20, help="early stopping patience (epochs)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lr-patience", type=int, default=7, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args()


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for (x,) in loader:
            loss = loss_fn(model(x), x)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


@torch.no_grad()
def reconstruct(model, spectra):
    model.eval()
    return model(torch.from_numpy(spectra)).numpy()


def plot_history(history, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.axvline(history["best_epoch"], color="grey", ls="--", lw=1, label="best")
    ax.set(xlabel="epoch", ylabel="MSE (z-normalised)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstructions(wavelength, x, x_hat, per_spec_mse, path, n=8):
    # Show spectra spanning the test-error distribution, from best to worst.
    order = np.argsort(per_spec_mse)
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    wl_order = np.argsort(wavelength)
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6), sharex=True)
    for ax, i in zip(axes.flat, picks):
        ax.plot(wavelength[wl_order], x[i, wl_order], "k.-", lw=1, ms=3, label="input")
        ax.plot(wavelength[wl_order], x_hat[i, wl_order], "C3-", lw=1.2, label="recon")
        ax.set_title(f"MSE={per_spec_mse[i]:.2e}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength [µm]")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    splits, wavelength, data_info = prepare_splits(
        args.data_dir, args.min_feature_height_ppm, seed=args.seed)
    print(json.dumps(data_info, indent=2))
    np.savez(out / "split_indices.npz", **{k: s.index for k, s in splits.items()})

    train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(splits["val"], 4096, shuffle=False)
    test_loader = make_loader(splits["test"], 4096, shuffle=False)

    model = Autoencoder(len(wavelength), tuple(args.hidden), args.latent)
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_state, best_epoch, bad_epochs = float("inf"), None, -1, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or bad_epochs == 0:
            print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  "
                  f"lr {history['lr'][-1]:.1e}  {'*' if bad_epochs == 0 else ''}")
        if bad_epochs >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    history["best_epoch"] = best_epoch
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    test_loss = run_epoch(model, test_loader, loss_fn)
    x_test = splits["test"].spectra
    x_hat = reconstruct(model, x_test)
    per_spec_mse = ((x_hat - x_test) ** 2).mean(axis=1)

    metrics = {
        **data_info,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "best_val_mse": best_val,
        "test_mse": test_loss,
        "test_mse_median": float(np.median(per_spec_mse)),
        "test_mse_p95": float(np.percentile(per_spec_mse, 95)),
        "args": vars(args),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2))

    torch.save({"model_state": best_state, "args": vars(args), "n_in": len(wavelength)},
               out / "model.pt")
    (out / "history.json").write_text(json.dumps(history))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, out / "loss.png")
    plot_reconstructions(wavelength, x_test, x_hat, per_spec_mse, out / "reconstructions.png")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
