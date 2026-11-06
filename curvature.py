"""Step 4: curvature of the spectral manifold and its focal set.

Geometry. With J = dS/du and H = d2S/du2 (ppm, unit-box coordinates), the second fundamental form
is the normal part of H, N = P_N H with P_N = I - J (J^T J)^-1 J^T. In the Fisher eigenbasis scaled
to unit length (whitened: a unit step along axis k moves the spectrum by 1 ppm), it becomes 52
symmetric 9x9 matrices B_k = Lam^-1/2 V^T N_k V Lam^-1/2 (units 1/ppm). From these:

- normal curvature along a unit tangent u is the data-space vector B(u, u). Its largest length
  kappa_max is the largest principal curvature, and rho_c = 1 / kappa_max (ppm) is the distance
  from the surface to its nearest focal point (centre of curvature). Along Fisher eigendirection k
  the radius is rho_k = 1 / |B_kk|.
- whether that bending is visible: over the surface's full width W_k along direction k (step 3,
  box face to box face), a circle of radius rho_k departs from its chord by the sagitta
  W_k^2 / (8 rho_k). Where that is below sigma, the bending in that direction is invisible to the
  data at noise sigma, however small rho_k is.
- intrinsic curvature (Gauss equation, exact in flat data space): sectional curvature
  K(a, b) = B_aa . B_bb - |B_ab|^2 and scalar curvature R = |sum_a B_aa|^2 - sum_ab |B_ab|^2 (1/ppm^2).
- focal set. For data y = S(theta) + eps with eps normal to the surface, theta is a critical point
  of chi^2 with Hessian J^T J - eps . H, i.e. I - sum_k eps_k B_k in this frame. theta stops being
  a minimum, and the minimum splits (a fold or cusp of the distance-squared function), exactly
  when lambda_max(sum_k eps_k B_k) > 1: the data have crossed the focal set. For white noise sigma,
  P_focal(sigma) = P(lambda_max(sum_k z_k B_k) > 1 / sigma), z ~ N(0, I), is the probability that
  noise carries the data across the focal set near the true planet.
- Bates & Watts (1980) intrinsic nonlinearity: gamma = kappa_max sigma sqrt(p); the Gaussian
  (linearised) confidence region is adequate when gamma < 1 / (2 sqrt(F(p, n - p; 0.95))).
- "data-constrained" (dc) variants restrict the tangent directions to the Fisher eigendirections
  the data constrain better than the prior (lam_k > 12 sigma^2 in the unit box).

Second derivatives are where an emulator is weakest, and whitening amplifies their errors along
sloppy directions. Every quantity is therefore computed for the ensemble mean and for each member,
and a planet counts as reliable for that quantity when the members agree within a factor 1.5.

The last part checks the focal-set prediction directly: for the most curved reliable interior
planets, data are placed at t x rho_c along the most curved normal, and the exact chi^2 Hessian and
the minima found by descent are compared with the prediction (the minimum splits at t = 1).

    python fisher.py && python curvature.py      # ~20 min on 4 cores; writes runs/curvature/
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
from scipy.stats import f as f_dist

from geometry import (CODE_COLOR, CODE_LABEL, CODES, PARAM_NAMES, box, derivatives, eigh_desc, eval_points,
                      frozen_ensemble, load_data)
from mode_paths import Chi2, descend

PRIOR_PRECISION = 12.0
N_BINS, N_PAR = 52, 9


def shape_operators(J, H):
    """J [n, 52, 9], H [n, 52, 9, 9] -> Fisher eigenvalues lam [n, 9] (descending), V [n, 9, 9],
    B [n, 52, 9, 9] in the whitened Fisher eigenframe (torch, float64)."""
    J, H = torch.as_tensor(J), torch.as_tensor(H)
    lam, V = eigh_desc((J.transpose(1, 2) @ J).numpy())
    lam, V = torch.as_tensor(lam.copy()), torch.as_tensor(V.copy())
    Q = torch.linalg.qr(J).Q  # orthonormal tangent basis
    N = H - torch.einsum("nkt,nlt,nlab->nkab", Q, Q, H)
    W = V / lam.clamp_min(1e-300).sqrt()[:, None, :]
    B = torch.einsum("nja,nkjl,nlb->nkab", W, N, W)
    return lam, V, B


def kappa_max(B, mask=None, n_random=8, iters=80, seed=0):
    """Largest normal curvature max_|u|=1 |B(u, u)| by alternating maximisation over the normal
    direction and the tangent direction, from the 9 axes plus random starts. Returns kappa [n],
    u* [n, 9] (whitened Fisher frame), unit normal n* [n, 52]."""
    n, _, d, _ = B.shape
    if mask is not None:
        m = mask.to(B.dtype)
        B = B * m[:, None, :, None] * m[:, None, None, :]
    g = torch.Generator().manual_seed(seed)
    u = torch.cat([torch.eye(d, dtype=B.dtype).expand(n, d, d),
                   torch.nn.functional.normalize(torch.randn(n, n_random, d, generator=g, dtype=B.dtype), dim=-1)], 1)
    if mask is not None:
        u = torch.nn.functional.normalize(u * m[:, None, :] + 1e-30, dim=-1)
    for _ in range(iters):
        b = torch.einsum("nkab,nra,nrb->nrk", B, u, u)
        nv = torch.nn.functional.normalize(b, dim=-1)
        M = torch.einsum("nrk,nkab->nrab", nv, B)
        u = torch.linalg.eigh(M)[1][..., -1]
    b = torch.einsum("nkab,nra,nrb->nrk", B, u, u)
    kap = b.norm(dim=-1)
    best = kap.argmax(1)
    ar = torch.arange(n)
    return kap[ar, best], u[ar, best], torch.nn.functional.normalize(b[ar, best], dim=-1)


def focal_lambda(B, n_mc, mask=None, seed=0, chunk=100):
    """lambda_max(sum_k z_k B_k) for n_mc standard-normal noise draws z: [n, n_mc] (1/ppm)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_mc, B.shape[1], generator=g, dtype=B.dtype)
    out = []
    for a in range(0, len(B), chunk):
        b = B[a:a + chunk]
        if mask is not None:
            m = mask[a:a + chunk].to(B.dtype)
            b = b * m[:, None, :, None] * m[:, None, None, :]
        out.append(torch.linalg.eigvalsh(torch.einsum("mk,nkab->nmab", z, b))[..., -1])
    return torch.cat(out)


def intrinsic(B):
    """Scalar curvature [n] and sectional curvatures K[a, b] [n, 9, 9] (1/ppm^2)."""
    diag = torch.diagonal(B, dim1=-2, dim2=-1)  # [n, 52, 9]: B_aa
    K = torch.einsum("nka,nkb->nab", diag, diag) - (B**2).sum(1)
    h = diag.sum(-1)
    R = (h**2).sum(-1) - (B**2).sum((1, 2, 3))
    return R, K



def curvature_set(lam_u, B, sigmas, n_mc, seed):
    """All curvature quantities for one model (ensemble mean or a member)."""
    kap, ustar, nstar = kappa_max(B, seed=seed)
    R, K = intrinsic(B)
    out = dict(kappa=kap.numpy(), ustar=ustar.numpy(), nstar=nstar.numpy(), scalar_R=R.numpy(),
               K12=K[:, 0, 1].numpy(), K89=K[:, 7, 8].numpy(), Kmax_abs=K.abs().amax((1, 2)).numpy(),
               kdir=torch.diagonal(B, dim1=-2, dim2=-1).norm(dim=1).numpy(),  # |B_kk|, [n, 9]
               focal_lambda=focal_lambda(B, n_mc, seed=seed + 1).float().numpy())
    for s in sigmas:
        mask = lam_u > PRIOR_PRECISION * s**2
        k, u, nv = kappa_max(B, mask=mask, seed=seed)
        out[f"kappa_dc_{s:g}"], out[f"ustar_dc_{s:g}"], out[f"nstar_dc_{s:g}"] = k.numpy(), u.numpy(), nv.numpy()
        out[f"focal_lambda_dc_{s:g}"] = focal_lambda(B, n_mc, mask=mask, seed=seed + 1).float().numpy()
        out[f"n_dc_{s:g}"] = mask.sum(1).numpy()
    return out


def member_ratio(members, key):
    x = np.stack([m[key] for m in members]).clip(1e-300)
    return x.max(0) / x.min(0)


def analyse(code, idx, lo, hi, args):
    t0 = time.time()
    models = frozen_ensemble(code, args.emulator_dir)
    S, P, _, _ = load_data(code, args.data_dir)
    theta = P[idx]
    Sm, Jm, Hm = derivatives(models, theta, lo, hi, hessian=True)
    print(f"[{code}] derivatives {time.time() - t0:.0f}s", flush=True)
    res = {"theta": theta, "feat_ppm": (S[idx].max(1) - S[idx].min(1)) * 1e6, "S_mean": Sm.mean(0)}
    lam, V, B = shape_operators(Jm.mean(0), Hm.mean(0))
    res["lam_u"], res["V_u"] = lam.numpy(), V.numpy()
    res["mean"] = curvature_set(lam, B, args.dc_sigmas, args.n_mc, args.seed)
    members = []
    for i in range(len(models)):
        lam_i, _, B_i = shape_operators(Jm[i], Hm[i])
        members.append(curvature_set(lam_i, B_i, args.dc_sigmas, args.n_mc // 2, args.seed))
    res["ratio"] = {k: member_ratio(members, k) for k in ["kappa", "kdir", "Kmax_abs"] +
                    [f"kappa_dc_{s:g}" for s in args.dc_sigmas]}
    res["rel"] = {k: v < args.reliable_ratio for k, v in res["ratio"].items()}
    # widths along the same (mean-model, unit-box) Fisher eigendirections, from step 3
    fz = np.load(os.path.join(args.fisher_dir, f"{code}_fisher.npz"))
    pos = np.searchsorted(fz["index"], idx)
    assert np.array_equal(fz["index"][pos], idx), "curvature points must be a subset of the step-3 points"
    res["width"] = fz["width_arc"][pos]
    print(f"[{code}] curvature {time.time() - t0:.0f}s", flush=True)
    res["validation"] = validate(models, res, lo, hi, args)
    print(f"[{code}] validation {time.time() - t0:.0f}s", flush=True)
    return res


def validate(models, res, lo, hi, args):
    """Place data at t x rho_c along the most curved normal and check where the minimum splits."""
    u0_all = (res["theta"] - lo) / (hi - lo)
    interior = ((u0_all > 0.05) & (u0_all < 0.95)).all(1)
    lo_t, span = torch.as_tensor(lo), torch.as_tensor(hi - lo)
    ts = np.array(args.validate_t)
    s_dc = f"{args.dc_sigmas[0]:g}"
    kinds = {"all directions": ("kappa", "ustar", "nstar"),
             f"data-constrained (sigma = {s_dc} ppm)": (f"kappa_dc_{s_dc}", f"ustar_dc_{s_dc}", f"nstar_dc_{s_dc}")}
    rows = []
    for kind, (kk, uk, nk) in kinds.items():
        kap = res["mean"][kk]
        cand = np.flatnonzero(res["rel"][kk] & interior & (kap > 0))
        order = cand[np.argsort(1 / kap[cand])]
        pick = list(order[:args.n_validate]) + list(order[len(order) // 2: len(order) // 2 + 1])
        for q in pick:
            rho = 1 / kap[q]
            u0 = torch.as_tensor(u0_all[q])
            lam, V = torch.as_tensor(res["lam_u"][q]), torch.as_tensor(res["V_u"][q])
            W = V / lam.sqrt()[None, :]
            ust = torch.as_tensor(res["mean"][uk][q])
            vstar = W @ ust
            vstar = vstar / vstar.norm()
            data = np.stack([(res["S_mean"][q] + t * rho * res["mean"][nk][q]) * 1e-6 for t in ts])
            hmin, hray = [], []
            for y in data:
                yt = torch.as_tensor(y * 1e6)
                fchi = lambda u: 0.5 * ((torch.stack([m(lo_t + u * span) for m in models]).mean(0) * 1e6 - yt) ** 2).sum()
                Hw = W.T @ torch.func.hessian(fchi)(u0) @ W
                hmin.append(torch.linalg.eigvalsh(Hw)[0].item())
                hray.append((ust @ Hw @ ust).item())
            f = Chi2(models, lo, hi, data, 1e-6)  # chi^2 in ppm^2
            starts = u0[None].expand(len(ts), 9)
            up = descend(f, (starts + args.validate_step * vstar).clamp(0, 1), args.descent_steps)
            dn = descend(f, (starts - args.validate_step * vstar).clamp(0, 1), args.descent_steps)
            with torch.no_grad():
                c0 = f(starts[:, None])[:, 0]
                cu, cd = f(up[:, None])[:, 0], f(dn[:, None])[:, 0]
            for i, t in enumerate(ts):
                drop = float(c0[i] - min(cu[i], cd[i]))
                rows.append({
                    "kind": kind, "point": int(q), "rho_c_ppm": float(rho), "t": float(t),
                    "hessian_rayleigh_along_u_star": hray[i], "hessian_min_eig_whitened": hmin[i],
                    "predicted_rayleigh": float(1 - t),
                    "split_distance": float((up[i] - dn[i]).norm()),
                    "chi2_drop_ppm2": drop, "delta_chi2_at_30ppm": drop / 900.0,
                    "minima_on_box_face": bool(((up[i] <= 1e-9) | (up[i] >= 1 - 1e-9) |
                                                (dn[i] <= 1e-9) | (dn[i] >= 1 - 1e-9)).any()),
                    "vstar_unit_box": dict(zip(PARAM_NAMES, vstar.numpy().round(3).tolist())),
                })
    return rows


def pctl(x, q=(5, 16, 50, 84, 95)):
    return np.percentile(x, q).tolist() if len(x) else []


def summarise(r, args):
    m, rel = r["mean"], r["rel"]
    p, n = N_PAR, N_BINS
    bw = 1 / (2 * np.sqrt(f_dist.ppf(0.95, p, n - p)))
    out = {"n_points": int(len(m["kappa"])), "percentiles": [5, 16, 50, 84, 95],
           "bates_watts_threshold_gamma": float(bw),
           "member_ratio_median": {k: (np.median(v, 0).round(3).tolist() if v.ndim > 1 else float(np.median(v)))
                                   for k, v in r["ratio"].items()},
           "frac_reliable": {k: (v.mean(0).round(3).tolist() if v.ndim > 1 else float(v.mean()))
                             for k, v in rel.items()}}
    # all directions
    rf = rel["kappa"]
    rho = 1 / m["kappa"]
    w = m["ustar"] ** 2
    full = {"rho_c_ppm_all_planets": pctl(rho), "rho_c_ppm_reliable": pctl(rho[rf]),
            "most_curved_direction_weight_on_v_k_median": np.median(w, 0).round(3).tolist(),
            "most_curved_direction_dominant_k_counts": np.bincount(w.argmax(1), minlength=9).tolist(),
            "frac_bates_watts_nonlinear_reliable": {f"{s:g}": float((m["kappa"][rf] * s * np.sqrt(p) > bw).mean())
                                                    for s in args.sigmas},
            "P_focal_reliable": {}}
    for s in args.sigmas:
        P_ = (m["focal_lambda"][rf] > 1 / s).mean(1)
        full["P_focal_reliable"][f"{s:g}"] = {"mean": float(P_.mean()), "median": float(np.median(P_)),
                                              "frac_planets_P_gt_0.05": float((P_ > 0.05).mean())}
    out["all_directions"] = full
    # per direction: radius, width, sagitta; reliability per direction
    rk = 1 / m["kdir"].clip(1e-300)
    W = r["width"]
    sag = W**2 / (8 * rk)
    rd = rel["kdir"]
    out["per_direction"] = {
        "rho_k_ppm_median": [float(np.median(rk[rd[:, k], k])) if rd[:, k].any() else None for k in range(9)],
        "width_ppm_median": np.median(W, 0).round(2).tolist(),
        "log10_rho_over_width_median": [float(np.median(np.log10(rk[rd[:, k], k] / W[rd[:, k], k])))
                                        if rd[:, k].any() else None for k in range(9)],
        "sagitta_ppm_median": [float(np.median(sag[rd[:, k], k])) if rd[:, k].any() else None for k in range(9)],
        "frac_sagitta_above_sigma": {f"{s:g}": [float((sag[rd[:, k], k] > s).mean()) if rd[:, k].any() else None
                                                for k in range(9)] for s in args.sigmas},
        "frac_reliable": rd.mean(0).round(3).tolist(),
    }
    # data-constrained directions
    out["data_constrained"] = {}
    for s in args.dc_sigmas:
        key = f"kappa_dc_{s:g}"
        rr = rel[key] & (m[key] > 0)
        P_ = (m[f"focal_lambda_dc_{s:g}"][rr] > 1 / s).mean(1)
        out["data_constrained"][f"{s:g}"] = {
            "n_directions_median": float(np.median(m[f"n_dc_{s:g}"])),
            "frac_reliable": float(rr.mean()),
            "rho_c_ppm_reliable": pctl(1 / m[key][rr]),
            "rho_c_over_sigma_median": float(np.median(1 / m[key][rr] / s)),
            "P_focal_mean": float(P_.mean()), "frac_planets_P_gt_0.05": float((P_ > 0.05).mean()),
            "frac_planets_P_gt_0.01": float((P_ > 0.01).mean()),
            "frac_bates_watts_nonlinear": float((m[key][rr] * s * np.sqrt(p) > bw).mean()),
            "most_curved_direction_weight_on_v_k_median": np.median(m[f"ustar_dc_{s:g}"][rr] ** 2, 0).round(3).tolist(),
        }
    # intrinsic curvature
    rK = rel["Kmax_abs"]
    out["intrinsic"] = {key: {"percentiles_16_50_84_per_ppm2": pctl(m[key][rK], (16, 50, 84)),
                              "frac_positive": float((m[key][rK] > 0).mean())}
                        for key in ("scalar_R", "K12", "K89")}
    out["intrinsic"]["radius_ppm_from_max_sectional_median"] = float(np.median(1 / np.sqrt(m["Kmax_abs"][rK])))
    out["intrinsic"]["radius_ppm_stiffest_plane_median"] = float(np.median(1 / np.sqrt(np.abs(m["K12"]).clip(1e-300))))
    out["intrinsic"]["frac_reliable"] = float(rK.mean())
    # feature dependence (data-constrained, first sigma)
    s0 = args.dc_sigmas[0]
    key = f"kappa_dc_{s0:g}"
    fb = np.array(args.feature_bins)
    kk = np.digitize(r["feat_ppm"], fb)
    out[f"dc_{s0:g}_by_feature_height"] = {}
    for i in range(len(fb) + 1):
        sel = (kk == i) & rel[key] & (m[key] > 0)
        if sel.sum() >= 20:
            lab = f"<{fb[0]:g}" if i == 0 else (f">{fb[-1]:g}" if i == len(fb) else f"{fb[i - 1]:g}-{fb[i]:g}")
            out[f"dc_{s0:g}_by_feature_height"][lab + " ppm"] = {
                "n": int(sel.sum()), "median_rho_c_ppm": float(np.median(1 / m[key][sel])),
                "P_focal_mean": float((m[f"focal_lambda_dc_{s0:g}"][sel] > 1 / s0).mean())}
    v = r["validation"]
    out["validation"] = v
    summ = {}
    for kind in sorted({x["kind"] for x in v}):
        rows = [x for x in v if x["kind"] == kind]
        summ[kind] = {
            "n_planets": len({x["point"] for x in rows}),
            "max_abs_rayleigh_error": float(max(abs(x["hessian_rayleigh_along_u_star"] - x["predicted_rayleigh"])
                                                for x in rows)),
            "median_split_distance_below_t1": float(np.median([x["split_distance"] for x in rows if x["t"] < 1])),
            "median_split_distance_above_t1": float(np.median([x["split_distance"] for x in rows if x["t"] > 1])),
            "max_delta_chi2_at_30ppm": float(max(x["delta_chi2_at_30ppm"] for x in rows)),
        }
    out["validation_summary"] = summ
    return out


def compare_codes(res, args):
    out = {}
    for a, b in (("pos", "prt"), ("pos", "taurex"), ("prt", "taurex")):
        if a in res and b in res:
            e = {}
            for key in ["kappa"] + [f"kappa_dc_{s:g}" for s in args.dc_sigmas]:
                rel = res[a]["rel"][key] & res[b]["rel"][key] & (res[a]["mean"][key] > 0) & (res[b]["mean"][key] > 0)
                d = np.abs(np.log10(res[a]["mean"][key][rel] / res[b]["mean"][key][rel]))
                e[key] = {"abs_dlog10_median": float(np.median(d)), "abs_dlog10_p90": float(np.percentile(d, 90)),
                          "n": int(rel.sum())}
            out[f"{a}-{b}"] = e
    return out


def plots(res, summ, args):
    codes = list(res)
    p = N_PAR
    bw = summ[codes[0]]["bates_watts_threshold_gamma"]
    s0 = args.dc_sigmas[0]

    # 1. focal distance: all directions vs data-constrained
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(-3, 5, 49)
    for c in codes:
        m, rel = res[c]["mean"], res[c]["rel"]
        ax.hist(np.log10(1 / m["kappa"][rel["kappa"]]), bins, histtype="step", lw=1, ls="--", color=CODE_COLOR[c],
                label=f"{CODE_LABEL[c]}: all directions")
        k = m[f"kappa_dc_{s0:g}"]
        rr = rel[f"kappa_dc_{s0:g}"] & (k > 0)
        ax.hist(np.clip(np.log10(1 / k[rr]), -3, 4.99), bins, histtype="step", lw=1.8, color=CODE_COLOR[c],
                label=f"{CODE_LABEL[c]}: data-constrained directions (σ = {s0:g})")
    ax.axvline(np.log10(s0), color="k", lw=0.9, label=f"σ = {s0:g} ppm")
    ax.axvline(np.log10(s0 * np.sqrt(p) / bw), color="grey", lw=0.9, label=f"Bates–Watts limit at σ = {s0:g} ppm")
    ax.set_xlabel("log₁₀ distance to the nearest focal point ρ_c = 1/κ_max (ppm)")
    ax.set_ylabel("planets (reliable)")
    ax.set_title("Distance from the surface of spectra to its focal set", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "focal_distance.png"), dpi=150)
    plt.close(fig)

    # 2. per-direction radius vs width (sagitta test)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    k = np.arange(1, 10)
    for i, c in enumerate(codes):
        m, rd = res[c]["mean"], res[c]["rel"]["kdir"]
        rk, W = 1 / m["kdir"].clip(1e-300), res[c]["width"]
        x = k + (i - 1) * 0.22
        for j in range(9):
            sel = rd[:, j]
            if sel.sum() < 10:
                continue
            q = np.percentile(np.log10(rk[sel, j]), [16, 50, 84])
            axs[0].vlines(x[j], q[0], q[2], color=CODE_COLOR[c], lw=4, alpha=0.6)
            axs[0].plot(x[j], q[1], "_", color="k", ms=7)
            qs = np.percentile(np.log10(W[sel, j] ** 2 / (8 * rk[sel, j])), [16, 50, 84])
            axs[1].vlines(x[j], qs[0], qs[2], color=CODE_COLOR[c], lw=4, alpha=0.6)
            axs[1].plot(x[j], qs[1], "_", color="k", ms=7)
        axs[0].plot(x, np.log10(np.median(W, 0)), "o", mfc="none", color=CODE_COLOR[c], ms=5)
        axs[0].plot([], [], color=CODE_COLOR[c], lw=4, label=f"{CODE_LABEL[c]}: radius ρ_k (bars), width (circles)")
        axs[1].plot([], [], color=CODE_COLOR[c], lw=4, label=CODE_LABEL[c])
    for ax in axs:
        ax.set_xticks(k, [f"v{j}" for j in k])
        ax.set_xlabel("unit-box Fisher eigendirection (stiffest → sloppiest)")
    for s, ls in zip((30, 50), ("--", ":")):
        axs[1].axhline(np.log10(s), color="k", lw=0.8, ls=ls, label=f"σ = {s} ppm")
    axs[0].set_ylabel("log₁₀ ppm")
    axs[0].set_title("Normal curvature radius along each direction vs the surface's width", fontsize=9)
    axs[1].set_ylabel("log₁₀ sagitta W²/(8ρ) (ppm)")
    axs[1].set_title("How far the surface bends away from straight across its full width", fontsize=9)
    axs[0].legend(fontsize=6.5)
    axs[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "radius_vs_width.png"), dpi=150)
    plt.close(fig)

    # 3. P_focal vs sigma
    sig = np.geomspace(3, 300, 30)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for c in codes:
        m, rel = res[c]["mean"], res[c]["rel"]
        lf = m["focal_lambda"][rel["kappa"]]
        P_ = np.stack([(lf > 1 / s).mean(1) for s in sig])
        axs[0].plot(sig, P_.mean(1), color=CODE_COLOR[c], ls="--", label=f"{CODE_LABEL[c]}: all directions")
        for s, mk in zip(args.dc_sigmas, ("o", "s")):
            key = f"kappa_dc_{s:g}"
            rr = rel[key] & (m[key] > 0)
            Pd = (m[f"focal_lambda_dc_{s:g}"][rr] > 1 / s).mean(1)
            axs[0].plot(s, Pd.mean(), mk, color=CODE_COLOR[c], label=f"{CODE_LABEL[c]}: data-constrained, σ = {s:g}")
            axs[1].hist(np.log10(1 / m[key][rr] / s), np.linspace(-1, 4, 41), histtype="step",
                        color=CODE_COLOR[c], ls="-" if mk == "o" else ":",
                        label=f"{CODE_LABEL[c]}, σ = {s:g}")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("noise σ per bin (ppm)")
    axs[0].set_ylabel("P(noise crosses the focal set), mean over planets")
    axs[0].set_yscale("symlog", linthresh=1e-3)
    axs[0].legend(fontsize=6)
    axs[1].axvline(0, color="k", lw=0.8)
    axs[1].axvline(np.log10(np.sqrt(p) / bw), color="grey", lw=0.8, label="Bates–Watts limit")
    axs[1].set_xlabel("log₁₀ (ρ_c / σ), data-constrained directions")
    axs[1].set_ylabel("planets (reliable)")
    axs[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "focal_probability.png"), dpi=150)
    plt.close(fig)

    # 4. validation
    kinds = sorted({x["kind"] for x in res[codes[0]]["validation"]})
    fig, axs = plt.subplots(len(kinds), len(codes), figsize=(4.4 * len(codes), 3.6 * len(kinds)), squeeze=False)
    for col, c in enumerate(codes):
        for row, kind in enumerate(kinds):
            ax = axs[row, col]
            rows = [x for x in res[c]["validation"] if x["kind"] == kind]
            for j, q in enumerate(sorted({x["point"] for x in rows})):
                rr = [x for x in rows if x["point"] == q]
                t = [x["t"] for x in rr]
                ax.plot(t, [x["hessian_rayleigh_along_u_star"] for x in rr], "o", ms=4, color=f"C{j}",
                        label=f"#{q}: ρ_c = {rr[0]['rho_c_ppm']:.3g} ppm, max Δχ²₃₀ = "
                              f"{max(x['delta_chi2_at_30ppm'] for x in rr):.2g}")
            tt = np.linspace(0, max(args.validate_t), 10)
            ax.plot(tt, 1 - tt, "k-", lw=1, label="prediction 1 − t")
            ax.axvline(1, color="k", lw=0.6, ls=":")
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xlabel("data offset along most curved normal / ρ_c")
            ax.set_ylabel("χ² Hessian along u* (whitened)")
            ax.set_title(f"{CODE_LABEL[c]}, {kind}", fontsize=8)
            ax.legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "focal_validation.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", nargs="+", default=CODES)
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("--n-mc", type=int, default=512, help="noise draws per planet for P_focal")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[10, 20, 30, 50, 100])
    ap.add_argument("--dc-sigmas", nargs="+", type=float, default=[30, 50])
    ap.add_argument("--reliable-ratio", type=float, default=1.5)
    ap.add_argument("--n-validate", type=int, default=3)
    ap.add_argument("--validate-t", nargs="+", type=float, default=[0.5, 0.9, 1.1, 1.5, 2.5])
    ap.add_argument("--validate-step", type=float, default=0.01)
    ap.add_argument("--descent-steps", type=int, default=600)
    ap.add_argument("--feature-bins", nargs="+", type=float, default=[30, 100, 210, 500, 1000, 3000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emulator-dir", default="runs/emulator")
    ap.add_argument("--fisher-dir", default="runs/fisher")
    ap.add_argument("--out", default="runs/curvature")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    lo, hi = box(args.data_dir)
    # a random subset of the step-3 points (same planets for every code)
    idx = np.sort(np.random.default_rng(1).permutation(eval_points(4000, args.data_dir))[:args.n_points])
    res, summ = {}, {}
    for c in args.codes:
        res[c] = analyse(c, idx, lo, hi, args)
        summ[c] = summarise(res[c], args)
        m = res[c]["mean"]
        np.savez_compressed(
            os.path.join(args.out, f"{c}_curvature.npz"), index=idx, theta=res[c]["theta"], feat_ppm=res[c]["feat_ppm"],
            lam_u=res[c]["lam_u"], width=res[c]["width"],
            **{f"ratio_{k}": v for k, v in res[c]["ratio"].items()},
            **{k: v for k, v in m.items() if not k.startswith("focal_lambda")},
            **{f"{k}_quantiles": np.quantile(v, np.linspace(0, 1, 101), axis=1).T
               for k, v in m.items() if k.startswith("focal_lambda")})
    summ["code_comparison"] = compare_codes(res, args)
    summ["settings"] = vars(args)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    plots(res, summ, args)


if __name__ == "__main__":
    main()
