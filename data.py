"""Loading and preprocessing of the binned POS spectra.

Pipeline (in this order):
  1. Feature-height cut: keep spectra with max(spec) - min(spec) >= 210 ppm.
  2. Z-normalisation per spectrum: (spec - mean(spec)) / std(spec).
  3. Random train / val / test split.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class Split:
    spectra: np.ndarray  # z-normalised spectra, (N, n_wavelengths)
    params: np.ndarray  # matching planet parameters, (N, n_params)
    mean: np.ndarray  # per-spectrum mean before z-norm, (N,)
    std: np.ndarray  # per-spectrum std before z-norm, (N,)
    index: np.ndarray  # row index into the original (uncut) arrays, (N,)


def feature_height(spectra):
    return spectra.max(axis=1) - spectra.min(axis=1)


def znormalise(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    return (spectra - mean) / std, mean[:, 0], std[:, 0]


def load_raw(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    spectra = np.load(data_dir / "pos_spec_binned.npy")
    params = np.load(data_dir / "pos_params_clean.npy")
    wavelength = np.load(data_dir / "wavelength_binned.npy")
    assert spectra.shape[0] == params.shape[0]
    assert spectra.shape[1] == wavelength.shape[0]
    return spectra, params, wavelength


def prepare_splits(data_dir=DATA_DIR, min_feature_height_ppm=210.0,
                   val_frac=0.15, test_frac=0.15, seed=42):
    """Apply the feature-height cut, z-normalise, and split.

    Spectra are transit depths as fractions, so the cut in ppm is scaled by 1e-6.
    Returns (splits dict with keys 'train'/'val'/'test', wavelength, info dict).
    """
    spectra, params, wavelength = load_raw(data_dir)

    fh = feature_height(spectra)
    keep = np.flatnonzero(fh >= min_feature_height_ppm * 1e-6)
    spectra, params = spectra[keep], params[keep]

    spectra_z, mean, std = znormalise(spectra)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keep))
    n_val = int(round(val_frac * len(perm)))
    n_test = int(round(test_frac * len(perm)))
    parts = {
        "test": perm[:n_test],
        "val": perm[n_test:n_test + n_val],
        "train": perm[n_test + n_val:],
    }
    splits = {
        name: Split(spectra_z[idx].astype(np.float32), params[idx], mean[idx], std[idx], keep[idx])
        for name, idx in parts.items()
    }
    info = {
        "n_total": int(len(fh)),
        "n_kept": int(len(keep)),
        "min_feature_height_ppm": float(min_feature_height_ppm),
        **{f"n_{name}": int(len(idx)) for name, idx in parts.items()},
    }
    return splits, wavelength, info


def make_loader(split, batch_size, shuffle):
    x = torch.from_numpy(split.spectra)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle)
