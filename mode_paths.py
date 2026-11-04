"""Step 2b: are matched pairs separate posterior modes or one connected valley?

Input: the feature-rich matched pairs from step 1 (runs/close_pairs/{code}_pairs.npz)
and the emulator ensemble from step 2a (runs/emulator/{code}/seed*.pt).

For a pair (i, j) indistinguishable at noise sigma, treat the true spectrum S_i
as noiseless data and use the ensemble-mean emulator as the forward model:

    chi2(theta) = ||S_emu(theta) - S_i||^2 / sigma^2.

1. Descend chi2 from theta_i and from theta_j to their local minima a and b with
   Levenberg-Marquardt steps (projected onto the prior box).
2. Optimise a minimum-energy path between a and b with the string method. Each
   interior image takes the damped Gauss-Newton step restricted to the plane
   perpendicular to the path, and images are respaced to equal arc length after
   every step. Gauss-Newton steps matter here: the problem is sloppy (chi2 rises
   thousands of times faster along stiff directions than along the valley), and
   fixed-size gradient steps (Adam) hold images off the valley floor and invent
   barriers.
3. The barrier is Delta = max chi2 along the path - max(chi2(a), chi2(b)), i.e. how
   far the likelihood must drop to get from the shallower mode to the deeper one.
   Delta < 0.5: one basin; Delta >= 2: separate modes (the likelihood dips by at
   least a factor e between them).

With --noisy the data are S_i plus one Gaussian noise draw, which moves them off
the manifold, as in a real observation. A secondary minimum counts as credible
when its chi2 is within 16.9 of the better minimum.

Pairs are used only if the ensemble mean reproduces both true spectra S_i and
S_j with chi2 < 2 at that sigma, so emulator error cannot create or hide a
barrier of the size that matters.

Any path's maximum is an upper bound on the true barrier, so an unconverged
string can only overstate a barrier. Two checks guard against that: the
barrier is re-evaluated with each ensemble member along the same path, and a
control set of pairs that are close by construction (theta_i plus a 0.05 random
step in the unit box) is run through the same procedure to measure the false
"separate mode" rate. A positive control pairs each reference with a partner
drawn uniformly from the prior box: it shows whether the chi2 landscape has
separate local minima at all and whether the method can resolve them.

    python mode_paths.py --codes pos --threads 1                  # noiseless data, one code
    python mode_paths.py --noisy --out runs/mode_paths_noisy      # one noise draw per pair
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from emulator import PARAM_NAMES, load_data, load_ensemble

CODE_LABEL = {"pos": "POS", "prt": "petitRADTRANS", "taurex": "TauREx"}
CHI2_95 = 16.919


class Chi2:
    """Batched chi2 over unit-box coordinates for a batch of data spectra."""

    def __init__(self, models, lo, hi, data, sigma):
        self.models, self.sigma = models, sigma
        self.lo = torch.as_tensor(lo)
        self.span = torch.as_tensor(hi - lo)
        self.data = torch.as_tensor(data)  # [B, 52]

    def spectra(self, u, members=False):
        th = self.lo + u * self.span
        preds = torch.stack([m(th) for m in self.models])  # [n_m, B, M, 52]
        return preds if members else preds.mean(0)

    def __call__(self, u, members=False):
        p = self.spectra(u, members)
        d = self.data[:, None, :]
        return ((p - d) ** 2).sum(-1) / self.sigma**2

    def _mean_spectrum(self, u1):
        th = self.lo + u1 * self.span
        return torch.stack([m(th) for m in self.models]).mean(0)

    def residual_jacobian(self, u):
        """r = (S_emu - data) / sigma, shape [B, M, 52], and dr/du, shape [B, M, 52, 9]."""
        B, M, D = u.shape
        flat = u.reshape(-1, D)
        with torch.no_grad():
            J = torch.func.vmap(torch.func.jacfwd(self._mean_spectrum))(flat)
            p = self._mean_spectrum(flat)
        r = (p.view(B, M, -1) - self.data[:, None, :]) / self.sigma
        return r, J.view(B, M, -1, D) / self.sigma


def lm_step(r, J, lam, u, tau=None, max_step=0.02):
    """Levenberg-Marquardt step for chi2 = |r|^2, optionally restricted to the subspace orthogonal to tau.

    Solves (Pi (A + lam D) Pi + tau tau^T) delta = -Pi g with A = J^T J, g = J^T r,
    D = diag(A), Pi = I - tau tau^T. The solution satisfies tau . delta = 0 and is the
    damped Gauss-Newton step within that subspace (tau = 0 gives the ordinary step).
    Coordinates sitting on a box face with the gradient pointing out of the box are
    held fixed (active set). The damping is raised until the step is at most
    max_step long (unit-box coordinates); any remaining excess is scaled away.
    """
    A = J.transpose(-1, -2) @ J
    g = (J.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
    # active set: coordinates on a box face with the gradient pushing outward are frozen
    free = ~(((u <= 0) & (g > 0)) | ((u >= 1) & (g < 0)))
    fm = free.to(A.dtype)
    A = A * fm.unsqueeze(-1) * fm.unsqueeze(-2) + torch.diag_embed(1 - fm)
    g = g * fm
    D = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(1e-12)
    eye = torch.eye(A.shape[-1], dtype=A.dtype)
    if tau is None:
        Pi, T = eye, 0.0
        rhs = -g
    else:
        T = tau.unsqueeze(-1) * tau.unsqueeze(-2)
        Pi = eye - T
        rhs = -(Pi @ g.unsqueeze(-1)).squeeze(-1)
    # trust region: raise the damping until the step is at most max_step long. Larger
    # damping suppresses the sloppy directions first, so descents reach the valley floor
    # before moving along it, and cannot jump across a barrier into another basin.
    lam = lam.clone()
    for _ in range(12):
        H = Pi @ (A + lam.unsqueeze(-1).unsqueeze(-1) * torch.diag_embed(D)) @ Pi + T + 1e-10 * eye
        delta = torch.linalg.solve(H, rhs) * fm
        n = delta.norm(dim=-1)
        over = n > max_step
        if not over.any():
            break
        lam = torch.where(over, lam * 4, lam)
    n = delta.norm(dim=-1, keepdim=True)
    return delta * (max_step / n.clamp_min(max_step))


def descend(f, u0, steps, lam0=1e-2):
    """Levenberg-Marquardt descent of chi2 from u0 [B, 9], projected onto the unit box."""
    u = u0[:, None].clone()
    lam = torch.full(u.shape[:2], lam0, dtype=u.dtype)
    r, J = f.residual_jacobian(u)
    c = (r**2).sum(-1)
    for _ in range(steps):
        un = (u + lm_step(r, J, lam, u)).clamp(0.0, 1.0)
        rn, Jn = f.residual_jacobian(un)
        cn = (rn**2).sum(-1)
        acc = cn < c
        a4 = acc.unsqueeze(-1)
        u = torch.where(a4, un, u)
        r = torch.where(a4, rn, r)
        J = torch.where(a4.unsqueeze(-1), Jn, J)
        c = torch.where(acc, cn, c)
        lam = torch.where(acc, (lam / 3).clamp_min(1e-7), (lam * 4).clamp_max(1e7))
    return u[:, 0]


def respace(u):
    """Redistribute images of each path [B, M, 9] to equal arc length."""
    B, M, _ = u.shape
    seg = (u[:, 1:] - u[:, :-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(B, 1, dtype=u.dtype), seg.cumsum(1)], 1)
    s = s / s[:, -1:].clamp_min(1e-12)
    t = torch.linspace(0, 1, M, dtype=u.dtype).expand(B, M).contiguous()
    k = torch.searchsorted(s.contiguous(), t, right=True).clamp(1, M - 1)
    s0, s1 = s.gather(1, k - 1), s.gather(1, k)
    w = ((t - s0) / (s1 - s0).clamp_min(1e-12)).unsqueeze(-1)
    idx0 = (k - 1).unsqueeze(-1).expand(-1, -1, u.shape[-1])
    idx1 = k.unsqueeze(-1).expand(-1, -1, u.shape[-1])
    return u.gather(1, idx0) * (1 - w) + u.gather(1, idx1) * w


def upsample(u, factor):
    B, M, D = u.shape
    t = torch.linspace(0, M - 1, (M - 1) * factor + 1, dtype=u.dtype)
    k = t.floor().long().clamp(max=M - 2)
    w = (t - k).view(1, -1, 1)
    return u[:, k] * (1 - w) + u[:, k + 1] * w


def string_method(f, ua, ub, n_images, iters, lam0=1e-2):
    """String method with Levenberg-Marquardt image updates.

    Interior images take the damped Gauss-Newton step restricted to the hyperplane
    perpendicular to the local path tangent; endpoints take the full step (free ends).
    A step is kept only if it lowers that image's chi2. Images are respaced to equal
    arc length after every iteration.
    """
    t = torch.linspace(0, 1, n_images, dtype=ua.dtype).view(1, -1, 1)
    u = (ua[:, None] * (1 - t) + ub[:, None] * t).clone()
    lam = torch.full(u.shape[:2], lam0, dtype=u.dtype)
    history = []
    for it in range(1, iters + 1):
        r, J = f.residual_jacobian(u)
        c = (r**2).sum(-1)
        tau = torch.zeros_like(u)
        tau[:, 1:-1] = u[:, 2:] - u[:, :-2]
        tau = tau / tau.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        un = (u + lm_step(r, J, lam, u, tau)).clamp(0.0, 1.0)
        with torch.no_grad():
            cn = f(un)
        acc = cn < c
        u = torch.where(acc.unsqueeze(-1), un, u)
        lam = torch.where(acc, (lam / 3).clamp_min(1e-7), (lam * 4).clamp_max(1e7))
        u = respace(u)
        if it % 10 == 0:
            history.append(torch.maximum(c, cn).max(1).values)
    return u, history


def run_batch(models, lo, hi, data, ua0, ub0, sigma, args):
    f = Chi2(models, lo, hi, data, sigma)
    ua = descend(f, ua0, args.descent_steps)
    ub = descend(f, ub0, args.descent_steps)
    path, hist = string_method(f, ua, ub, args.images, args.iters)
    fine = upsample(path, 4)
    with torch.no_grad():
        c_mean = f(fine)
        c_mem = f(fine, members=True)  # [n_m, B, L]
    ends = torch.maximum(c_mean[:, 0], c_mean[:, -1])
    barrier = (c_mean.max(1).values - ends).clamp_min(0)
    ends_m = torch.maximum(c_mem[..., 0], c_mem[..., -1])
    barrier_m = (c_mem.max(-1).values - ends_m).clamp_min(0)
    # convergence: highest image chi2 three quarters of the way through vs at the end
    half = hist[(3 * len(hist)) // 4] if hist else None
    return {
        "ua": path[:, 0].numpy(), "ub": path[:, -1].numpy(), "path": path.numpy(),
        "chi2_a": c_mean[:, 0].numpy(), "chi2_b": c_mean[:, -1].numpy(),
        "barrier": barrier.numpy(), "barrier_members": barrier_m.T.numpy(),
        "path_chi2": c_mean.numpy(),
        "max_chi2_halfway": half.numpy() if half is not None else np.full(len(ua), np.nan),
    }


def classify(r, args):
    d = np.linalg.norm(r["ua"] - r["ub"], axis=1)
    one = (r["barrier"] < args.one_basin) | (d < args.same_point)
    sep = (r["barrier"] >= args.sep) & ~one
    robust = sep & (r["barrier_members"].min(1) >= args.sep)
    return one, sep, robust, d


def credible(r):
    """Secondary minimum lies within the 95% region around the better minimum."""
    return np.abs(r["chi2_a"] - r["chi2_b"]) < CHI2_95


def run_code(code, args, rng):
    t0 = time.time()
    models = load_ensemble(code, args.emulator_dir)
    for m in models:
        for p in m.parameters():
            p.requires_grad_(False)
    S, P, _, valid = load_data(code, args.data_dir)
    lo, hi = P.min(0), P.max(0)
    U = (P - lo) / (hi - lo)
    pairs = np.load(os.path.join(args.pairs_dir, f"{code}_pairs.npz"))
    ok = valid[pairs["i"]] & valid[pairs["j"]]
    # emulator fidelity at the pair endpoints: squared error of the ensemble mean, per spectrum
    ends = np.unique(np.r_[pairs["i"], pairs["j"]])
    with torch.no_grad():
        th = torch.as_tensor(P[ends])
        pe = torch.stack([m(th) for m in models]).mean(0).numpy()
    emu_err2 = np.full(len(S), np.inf)
    emu_err2[ends] = ((pe - S[ends]) ** 2).sum(1)
    out = {"code": code, "n_members": len(models), "per_sigma": {}}
    store = {}
    for sigma in args.sigmas:
        s2 = (sigma * 1e-6) ** 2
        cand = ok & (pairs["sigma_min_ppm"] < sigma)
        faithful = (emu_err2[pairs["i"]] / s2 < args.max_emu_chi2) & (emu_err2[pairs["j"]] / s2 < args.max_emu_chi2)
        n_cand, n_faith = int(cand.sum()), int((cand & faithful).sum())
        sel = np.flatnonzero(cand & faithful)
        if len(sel) > args.max_pairs:
            sel = np.sort(rng.choice(sel, args.max_pairs, replace=False))
        I, J = pairs["i"][sel], pairs["j"][sel]
        # control: theta_i plus a small random step, clipped to the box
        nc = min(args.n_control, len(I))
        ci = rng.choice(I, nc, replace=False)
        step = rng.normal(size=(nc, 9))
        step = 0.05 * step / np.linalg.norm(step, axis=1, keepdims=True)
        uc = np.clip(U[ci] + step, 0, 1)
        # random: theta_i and a partner drawn uniformly from the prior box (probes the whole landscape)
        nr = min(args.n_random, len(I))
        ri = rng.choice(I, nr, replace=False)
        ur = rng.uniform(size=(nr, 9))

        res = {}
        sets = {"pairs": (I, U[I], U[J]), "control": (ci, U[ci], uc), "random": (ri, U[ri], ur)}
        for name, (ii, ua0, ub0) in sets.items():
            parts = []
            for a in range(0, len(ii), args.batch):
                sl = slice(a, a + args.batch)
                data = S[ii[sl]]
                if args.noisy:
                    data = data + rng.normal(scale=sigma * 1e-6, size=data.shape)
                parts.append(run_batch(models, lo, hi, data, torch.as_tensor(ua0[sl]),
                                       torch.as_tensor(ub0[sl]), sigma * 1e-6, args))
            res[name] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            res[name]["ref"] = ii
        res["pairs"]["partner"] = J

        e = {"n_matched_pairs": n_cand, "n_emulator_faithful": n_faith, "max_emu_chi2": args.max_emu_chi2}
        for name in ("pairs", "control", "random"):
            r = res[name]
            one, sep, robust, dist = classify(r, args)
            e[name] = {
                "n": int(len(one)),
                "frac_one_basin": float(one.mean()),
                "frac_separate_modes": float(sep.mean()),
                "frac_separate_modes_all_members": float(robust.mean()),
                "frac_ambiguous": float((~one & ~sep).mean()),
                "median_minima_distance": float(np.median(dist)),
                "median_chi2_secondary_minimum": float(np.median(np.maximum(r["chi2_a"], r["chi2_b"]))),
                "frac_separate_and_credible": float((sep & credible(r)).mean()),
                "barrier_pct_50_90_99": np.percentile(r["barrier"], [50, 90, 99]).round(3).tolist(),
                "frac_unconverged": float(np.mean(
                    np.abs(r["max_chi2_halfway"] - r["path_chi2"].max(1)) > 0.1 * r["path_chi2"].max(1) + 0.5)),
            }
            if name == "pairs":
                da = np.abs(r["ua"] - r["ub"])
                for lab, m in (("separate", sep), ("one_basin", one)):
                    if m.any():
                        e[name][f"median_abs_du_minima_{lab}"] = {p: float(np.median(da[m, q]))
                                                                  for q, p in enumerate(PARAM_NAMES)}
                if sep.any():
                    th_a, th_b = lo + r["ua"][sep] * (hi - lo), lo + r["ub"][sep] * (hi - lo)
                    dc = th_b[:, 4:9] - th_a[:, 4:9]
                    sw = (dc.max(1) >= 2) & (dc.min(1) <= -2)
                    e[name]["frac_separate_with_species_swap_2dex"] = float(sw.mean())
        out["per_sigma"][f"{sigma:g}"] = e
        store[sigma] = res
        np.savez_compressed(os.path.join(args.out, f"{code}_sigma{sigma:g}.npz"),
                            **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items() if k != "path"},
                            pairs_path=res["pairs"]["path"])
        print(f"[{code} σ={sigma:g}] pairs: one basin {e['pairs']['frac_one_basin']:.2f}, "
              f"separate {e['pairs']['frac_separate_modes']:.2f} (all members {e['pairs']['frac_separate_modes_all_members']:.2f}); "
              f"control separate {e['control']['frac_separate_modes']:.3f}; random separate "
              f"{e['random']['frac_separate_modes']:.2f}, credible {e['random']['frac_separate_and_credible']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out, store, (S, P, lo, hi)


def plots(results, args):
    codes = list(results)
    colors = {"pos": "#1f77b4", "prt": "#d62728", "taurex": "#2ca02c"}

    fig, axs = plt.subplots(1, len(args.sigmas), figsize=(5 * len(args.sigmas), 3.8), squeeze=False)
    bins = np.concatenate([[0, 0.5], np.logspace(np.log10(0.5), 4, 30)[1:]])
    for ax, sigma in zip(axs[0], args.sigmas):
        for c in codes:
            r = results[c][1][sigma]
            ax.hist(np.clip(r["pairs"]["barrier"], 0, 9999), bins=bins, histtype="step", lw=1.5,
                    color=colors[c], label=CODE_LABEL[c])
            ax.hist(np.clip(r["control"]["barrier"], 0, 9999), bins=bins, histtype="step", lw=1,
                    ls=":", color=colors[c])
            ax.hist(np.clip(r["random"]["barrier"], 0, 9999), bins=bins, histtype="step", lw=1,
                    ls="--", color=colors[c])
        ax.set_xscale("symlog", linthresh=0.5)
        ax.axvline(args.one_basin, color="k", lw=0.8)
        ax.axvline(args.sep, color="k", lw=0.8, ls="--")
        ax.set_xlabel("barrier Δχ² along optimised path")
        ax.set_ylabel("pairs")
        ax.set_title(f"σ = {sigma:g} ppm \n(solid: matched pairs, dotted: close control, dashed: random partner)", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "barrier_hist.png"), dpi=150)
    plt.close(fig)

    # parameter differences between minima: separate modes vs one basin
    sigma = args.sigmas[-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / (2 * len(codes))
    for k, c in enumerate(codes):
        e = results[c][0]["per_sigma"][f"{sigma:g}"]["pairs"]
        for h, (lab, hatch) in enumerate((("separate", ""), ("one_basin", "//"))):
            key = f"median_abs_du_minima_{lab}"
            if key not in e:
                continue
            x = np.arange(9) + (2 * k + h - len(codes) + 0.5) * w
            ax.bar(x, [e[key][p] for p in PARAM_NAMES], w, color=colors[c], hatch=hatch,
                   alpha=1 if h == 0 else 0.45,
                   label=f"{CODE_LABEL[c]} {'separate modes' if h == 0 else 'one basin'}")
    ax.set_xticks(range(9), PARAM_NAMES)
    ax.set_ylabel("median |Δu| between minima")
    ax.set_title(f"Where the two minima differ (σ = {sigma:g} ppm)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "minima_param_diffs.png"), dpi=150)
    plt.close(fig)

    # example paths: the deepest robust separate mode per code
    wl = np.load(os.path.join(args.data_dir, "wavelength_binned.npy"))
    fig, axs = plt.subplots(3, len(codes), figsize=(4.6 * len(codes), 9), squeeze=False)
    for k, c in enumerate(codes):
        out, store, (S, P, lo, hi) = results[c]
        r = store[sigma]["pairs"]
        one, sep, robust, _ = classify(r, args)
        cand = np.flatnonzero(robust & credible(r))
        if not len(cand):
            axs[0, k].set_title(f"{CODE_LABEL[c]}: no robust separate modes")
            continue
        q = cand[np.argmax(r["barrier"][cand])]
        pc = r["path_chi2"][q]
        s = np.linspace(0, 1, len(pc))
        axs[0, k].plot(s, pc, color=colors[c])
        axs[0, k].axhline(CHI2_95, color="k", lw=0.8, ls="--")
        axs[0, k].set_yscale("symlog", linthresh=1)
        axs[0, k].set_xlabel("position along path")
        axs[0, k].set_ylabel("χ²")
        axs[0, k].set_title(f"{CODE_LABEL[c]} ref #{r['ref'][q]}, σ={sigma:g} ppm: barrier Δχ²={r['barrier'][q]:.0f}",
                            fontsize=8)
        path = r["path"][q] if "path" in r else None
        if path is not None:
            for d in range(9):
                axs[1, k].plot(np.linspace(0, 1, len(path)), path[:, d], label=PARAM_NAMES[d])
        axs[1, k].set_xlabel("position along path")
        axs[1, k].set_ylabel("unit-box coordinate")
        axs[1, k].legend(fontsize=6, ncol=3)
        models = load_ensemble(c, args.emulator_dir)
        th = lo + np.stack([r["ua"][q], r["ub"][q]]) * (hi - lo)
        with torch.no_grad():
            sp = torch.stack([m(torch.as_tensor(th)) for m in models]).mean(0).numpy()
        axs[2, k].plot(wl, S[r["ref"][q]] * 1e6, "k-", lw=1.2, label="data (true spectrum)")
        axs[2, k].plot(wl, sp[0] * 1e6, "--", color=colors[c], label=f"mode a, χ²={r['chi2_a'][q]:.1f}")
        axs[2, k].plot(wl, sp[1] * 1e6, ":", color=colors[c], lw=2, label=f"mode b, χ²={r['chi2_b'][q]:.1f}")
        axs[2, k].set_xscale("log")
        axs[2, k].set_xlabel("wavelength (µm)")
        axs[2, k].set_ylabel("transit depth (ppm)")
        axs[2, k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "example_paths.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", nargs="+", default=["pos", "prt", "taurex"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[30, 50])
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--n-control", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--iters", type=int, default=120, help="string-method iterations")
    ap.add_argument("--descent-steps", type=int, default=300, help="Levenberg-Marquardt steps per endpoint")
    ap.add_argument("--one-basin", type=float, default=0.5, help="barrier below this: one basin")
    ap.add_argument("--same-point", type=float, default=0.02, help="minima closer than this (unit box): one basin")
    ap.add_argument("--sep", type=float, default=2.0, help="barrier at or above this: separate modes")
    ap.add_argument("--max-emu-chi2", type=float, default=2.0,
                    help="use a pair only if the emulator reproduces both true spectra with chi2 below this")
    ap.add_argument("--noisy", action="store_true",
                    help="data = true spectrum + one Gaussian noise draw at sigma (default: noiseless)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pairs-dir", default="runs/close_pairs")
    ap.add_argument("--emulator-dir", default="runs/emulator")
    ap.add_argument("--out", default="runs/mode_paths")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {c: run_code(c, args, rng) for c in args.codes}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({c: r[0] for c, r in results.items()} | {"args": vars(args)}, f, indent=2)
    plots(results, args)


if __name__ == "__main__":
    main()
