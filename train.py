"""Train the spectrum autoencoder on the binned spectra.

Usage:
    python train.py                       # defaults below
    python train.py --lr 5e-4 --out runs/lr5e-4
    python train.py --noise-ppm 20        # denoising: noisy inputs, clean targets

With --noise-ppm, white Gaussian noise of that many ppm is added to the raw spectra (a fresh
draw every batch for training, one fixed draw for val/test). The input is the noisy spectrum
z-normalised with its own mean and std; the target is the clean spectrum scaled by that same
mean and std, so output * std + mean is the denoised spectrum in transit-depth units.
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
from torch.utils.data import DataLoader, TensorDataset

from data import DATA_DIR, DATASETS, make_loader, prepare_splits
from model import ACTIVATIONS, Autoencoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="runs/ae_baseline")
    p.add_argument("--dataset", default="pos", choices=DATASETS)
    p.add_argument("--cut-datasets", nargs="+", choices=DATASETS, default=None,
                   help="keep only planets passing the feature-height cut in all of these (default: --dataset)")
    p.add_argument("--min-feature-height-ppm", type=float, default=210.0)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32, 16])
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--activation", default="gelu", choices=sorted(ACTIVATIONS))
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20, help="early stopping patience (epochs)")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="relative val-loss improvement needed to reset the early stopping patience")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"],
                   help="plateau: ReduceLROnPlateau; cosine: CosineAnnealingLR over --epochs")
    p.add_argument("--lr-patience", type=int, default=7, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6, help="final learning rate for the cosine schedule")
    p.add_argument("--seed", type=int, default=42, help="seed for weight init and batch order")
    p.add_argument("--split-seed", type=int, default=None,
                   help="seed for the train/val/test split (default: --seed)")
    p.add_argument("--noise-ppm", type=float, default=0.0,
                   help="train as a denoising AE with this white noise per bin (ppm); 0 = noiseless")
    p.add_argument("--threads", type=int, default=None)
    return p.parse_args()


def raw_spectra(split):
    """Transit-depth spectra (float64 tensor) recovered from a z-normalised split."""
    return torch.from_numpy(split.spectra.astype(np.float64) * split.std[:, None] + split.mean[:, None])


def noisy_pairs(raw, sigma, generator=None):
    """(input, target): noisy spectrum z-normalised by its own mean/std, clean spectrum scaled the same way."""
    noisy = raw + sigma * torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
    m = noisy.mean(1, keepdim=True)
    s = noisy.std(1, unbiased=False, keepdim=True)
    return ((noisy - m) / s).float(), ((raw - m) / s).float()


def run_epoch(model, loader, loss_fn, optimizer=None, noise=0.0):
    """Batches are (x,) for plain reconstruction, (raw,) with noise > 0, or fixed (input, target) pairs."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            if len(batch) == 2:
                x, target = batch
            elif noise:
                x, target = noisy_pairs(batch[0], noise)
            else:
                x = target = batch[0]
            loss = loss_fn(model(x), target)
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
        ax.plot(wavelength[wl_order], x[i, wl_order], "k.-", lw=1, ms=3, label="target")
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
        args.data_dir, args.min_feature_height_ppm,
        seed=args.seed if args.split_seed is None else args.split_seed,
        dataset=args.dataset, cut_datasets=args.cut_datasets)
    print(json.dumps(data_info, indent=2))
    np.savez(out / "split_indices.npz", **{k: s.index for k, s in splits.items()})

    noise = args.noise_ppm * 1e-6
    if noise:
        gen = torch.Generator().manual_seed(args.seed)
        fixed = {k: noisy_pairs(raw_spectra(splits[k]), noise, gen) for k in ("val", "test")}
        train_loader = DataLoader(TensorDataset(raw_spectra(splits["train"])), args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(*fixed["val"]), 4096)
        test_loader = DataLoader(TensorDataset(*fixed["test"]), 4096)
    else:
        train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
        val_loader = make_loader(splits["val"], 4096, shuffle=False)
        test_loader = make_loader(splits["test"], 4096, shuffle=False)

    model = Autoencoder(len(wavelength), tuple(args.hidden), args.latent, ACTIVATIONS[args.activation])
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_factor, patience=args.lr_patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    # best_val/best_state track the lowest val loss seen; ref_val is the last loss that beat
    # the previous reference by more than min_delta (relative) and drives early stopping.
    best_val, best_state, best_epoch = float("inf"), None, -1
    ref_val, bad_epochs = float("inf"), 0
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, noise)
        val_loss = run_epoch(model, val_loader, loss_fn)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if val_loss < ref_val * (1 - args.min_delta):
            ref_val, bad_epochs = val_loss, 0
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
    if noise:
        x_in, x_test = (t.numpy() for t in fixed["test"])
    else:
        x_in = x_test = splits["test"].spectra
    x_hat = reconstruct(model, x_in)
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
