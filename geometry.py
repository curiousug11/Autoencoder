"""Shared helpers for steps 3 and 4: evaluation points and emulator derivatives.

Coordinates
- unit box u in [0, 1]^9: each parameter scaled by its prior range. The priors are uniform in
  these coordinates (Rs, Rp, Mp, T uniform in linear units; chem1-5 uniform in log10 abundance),
  so this is the natural frame for prior-relative statements and for the box boundary.
- log coordinates phi: log10 Rs, log10 Rp, log10 Mp, log10 T, chem1..5 (already log10). This is
  the usual frame for sloppiness (relative changes), and in it the step-1 directions are exact:
  Rp/Rs is (-1, 1, 0, ...), the feature amplitude at fixed Rp/Rs is (1/2, 1/2, -1, 1, 0, ...), and
  scaling Rs, Rp and Mp together is (1, 1, 1, 0, ...). The three are mutually orthogonal.
Spectra and derivatives are in ppm.
"""
import numpy as np
import torch

from emulator import PARAM_NAMES, load_data, load_ensemble, split_indices

CODES = ["pos", "prt", "taurex"]
CODE_LABEL = {"pos": "POS", "prt": "petitRADTRANS", "taurex": "TauREx"}
CODE_COLOR = {"pos": "#1f77b4", "prt": "#d62728", "taurex": "#2ca02c"}
N_LIN = 4  # Rs, Rp, Mp, T are linear parameters; the rest are already log10

_s3 = np.sqrt
DIRECTIONS = {  # in log coordinates, unit length
    "Rp/Rs": np.array([-1, 1, 0, 0, 0, 0, 0, 0, 0]) / _s3(2),
    # the feature amplitude T Rp^3 / (Mp Rs^2) is (-2, 3, -1, 1, 0, ...), which is 83% Rp/Rs; this is
    # its part orthogonal to Rp/Rs, i.e. changing the amplitude at fixed Rp/Rs
    "amplitude at fixed Rp/Rs": np.array([0.5, 0.5, -1, 1, 0, 0, 0, 0, 0]) / _s3(2.5),
    "scale Rs,Rp,Mp": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]) / _s3(3),
}


def box(data_dir="data"):
    P = load_data("pos", data_dir)[1]
    return P.min(0), P.max(0)


def eval_points(n, data_dir="data", seed=0):
    """n test-split planets whose spectra are valid in every code (the same planets for all codes)."""
    valid = None
    for c in CODES:
        v = load_data(c, data_dir)[3]
        valid = v if valid is None else valid & v
    test = split_indices(len(valid))[2]
    test = test[valid[test]]
    return np.sort(np.random.default_rng(seed).choice(test, n, replace=False))


def log_factors(theta, lo, hi):
    """du/dphi per parameter, shape [n, 9]: multiply a unit-box Jacobian column by this for log coordinates."""
    c = np.tile(1.0 / (hi - lo), (len(theta), 1))
    c[:, :N_LIN] *= theta[:, :N_LIN] * np.log(10)
    return c


def frozen_ensemble(code, root="runs/emulator"):
    models = load_ensemble(code, root)
    for m in models:
        for p in m.parameters():
            p.requires_grad_(False)
    return models


def derivatives(models, theta, lo, hi, hessian=False, chunk=250):
    """Per member: spectrum [M, n, 52], Jacobian [M, n, 52, 9] and optionally Hessian [M, n, 52, 9, 9],
    all in ppm, with respect to unit-box coordinates."""
    lo_t, span = torch.as_tensor(lo), torch.as_tensor(hi - lo)
    u_all = (torch.as_tensor(theta) - lo_t) / span
    S, J, H = [], [], []
    for m in models:
        f = lambda u: m(lo_t + u * span) * 1e6
        s, j, h = [], [], []
        for a in range(0, len(u_all), chunk):
            u = u_all[a:a + chunk]
            with torch.no_grad():
                s.append(f(u))
            j.append(torch.func.vmap(torch.func.jacrev(f))(u))
            if hessian:
                h.append(torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(f)))(u))
        S.append(torch.cat(s))
        J.append(torch.cat(j))
        if hessian:
            H.append(torch.cat(h))
    out = [torch.stack(S).numpy(), torch.stack(J).numpy()]
    if hessian:
        out.append(torch.stack(H).numpy())
    return out


def eigh_desc(A):
    """Eigen-decomposition of symmetric [..., d, d], eigenvalues descending (stiffest first)."""
    w, V = np.linalg.eigh(A)
    return w[..., ::-1], V[..., ::-1]


def subspace_sin(Va, Vb, m):
    """Sine of the largest principal angle between the spans of the first m columns of Va and Vb."""
    A, B = Va[..., :m], Vb[..., :m]
    R = B - A @ (np.swapaxes(A, -1, -2) @ B)
    return np.linalg.norm(R, ord=2, axis=(-2, -1))


__all__ = ["CODES", "CODE_LABEL", "CODE_COLOR", "DIRECTIONS", "PARAM_NAMES", "box", "eval_points",
           "log_factors", "frozen_ensemble", "derivatives", "eigh_desc", "subspace_sin", "load_data"]
