"""Step 5: full posteriors on the emulators, tested against the geometry of steps 3 and 4.

Each retrieval is a simulated observation y = S(theta0) + noise of a test-set planet, with
S the ensemble-mean emulator (so emulator error cannot pose as multimodality: the test is of the
geometry measured in steps 3-4) and white noise sigma = 30 or 50 ppm per bin. The prior is uniform
on the unit box.

Sampler. Sequential Monte Carlo with adaptive likelihood tempering (e.g. Del Moral, Doucet & Jasra
2006): particles start from the prior, the inverse temperature beta rises from 0 to 1 in steps
chosen so the effective sample size stays at half the particles, with systematic resampling and
differential-evolution Metropolis moves (ter Braak 2006), which adapt to each mode's shape
however sloppy it is. Starting from the prior, every mode is populated in proportion to its mass, and the
evidence comes for free. All retrievals run together as one batched computation. The sampler is
checked against a toy with a known answer and against dynesty nested sampling (--validate).

Modes. Tempering can empty a minority mode before it separates (the toy in --validate shows it), so
modes are found by a census instead of read off the particles. Starting points are 16 final
posterior samples plus 24 particles saved at intermediate temperatures (beta = 1e-6 ... 0.1, spread
over every basin before any mode can be lost). Each is descended to its local chi^2 minimum
(Levenberg-Marquardt, step 2); the minima are grouped (within 0.03 in the unit box), and each
group's barrier to the best group is measured with the string method; any barrier >= 2 is
re-measured with re-descended ends and 12 routes (refine_modes.py). The mass of each separate mode
is estimated by importance sampling around it (multivariate-t proposal with the local Fisher +
prior covariance, inflated 1.5x; points are credited only to their nearest mode), relative to the
SMC evidence of the main mode. A retrieval is multimodal when separate modes (refined barrier >= 2)
hold at least 5% of the posterior, by either the SMC sample share or the importance-sampling mass.

Geometry, evaluated at the true planet before looking at the posterior:
- the focal-crossing statistic for the actual noise draw: Lambda = lambda_max(sum_k eps_k B_k),
  where eps is the noise and B the step-4 shape operators; Lambda > 1 means this noise carried
  the data across the focal set. For all directions and for the data-constrained ones.
- P_focal (the probability over noise draws), and the Bates-Watts intrinsic (gamma_N) and
  parameter-effects (gamma_T) nonlinearities of the data-constrained directions.
- the Fisher + uniform-prior Gaussian prediction of each parameter's posterior width (as in step 3).
Each posterior's marginal widths are compared with that prediction; the median |log ratio| over
the 9 parameters is tested against gamma_N and gamma_T.

    python posterior.py --validate        # sampler checks (~15 min)
    python posterior.py --controls        # pipeline controls with known bimodal / unimodal data
    python posterior.py                   # all retrievals (~3 h on 4 cores); writes runs/posterior/
    python posterior.py --plots-only
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from curvature import PRIOR_PRECISION, kappa_max, shape_operators
from geometry import CODE_COLOR, CODE_LABEL, CODES, PARAM_NAMES, box, derivatives, frozen_ensemble
from mode_paths import Chi2, descend, string_method, upsample
from refine_modes import refine_group

D = 9
DT = torch.float64


# ----------------------------------------------------------------------------- likelihoods
class EmuLike:
    """log L = -chi^2 / 2 for a batch of runs with the ensemble-mean emulator (ppm units)."""

    def __init__(self, models, lo, hi, data_ppm, sigma_ppm, chunk=120000):
        self.models, self.chunk = models, chunk
        self.lo, self.span = torch.as_tensor(lo), torch.as_tensor(hi - lo)
        self.data = torch.as_tensor(data_ppm)
        self.sig = torch.as_tensor(sigma_ppm, dtype=DT)
        self.n_eval = 0

    @torch.no_grad()
    def spectra(self, U):
        flat = U.reshape(-1, D)
        out = []
        for a in range(0, len(flat), self.chunk):
            th = self.lo + flat[a:a + self.chunk] * self.span
            out.append(torch.stack([m(th) for m in self.models]).mean(0) * 1e6)
        self.n_eval += len(flat)
        return torch.cat(out).view(*U.shape[:-1], -1)

    def __call__(self, X, runs):
        S = self.spectra(X)
        return -0.5 * ((S - self.data[runs][:, None, :]) ** 2).sum(-1) / self.sig[runs][:, None] ** 2


class ToyLike:
    """Two anisotropic Gaussians in the unit box (weights w, 1 - w); exact evidence 1 (log Z = 0)."""

    def __init__(self, R, w=0.7, seed=0):
        g = np.random.default_rng(seed)
        self.w = w
        self.mu = torch.as_tensor(np.stack([np.full(D, 0.3), np.full(D, 0.7)]) + g.uniform(-0.05, 0.05, (2, D)))
        prec, logdet = [], []
        for _ in range(2):
            Q = np.linalg.qr(g.normal(size=(D, D)))[0]
            s = np.geomspace(1e-4, 0.03, D)  # sloppy: widths over 2.5 decades
            C = Q @ np.diag(s**2) @ Q.T
            prec.append(np.linalg.inv(C))
            logdet.append(np.linalg.slogdet(C)[1])
        self.prec, self.logdet = torch.as_tensor(np.stack(prec)), torch.as_tensor(np.array(logdet))
        self.n_eval = 0

    def __call__(self, X, runs):
        self.n_eval += X.shape[0] * X.shape[1]
        lp = []
        for c, wc in enumerate((self.w, 1 - self.w)):
            d = X - self.mu[c]
            q = torch.einsum("rni,ij,rnj->rn", d, self.prec[c], d)
            lp.append(np.log(wc) - 0.5 * (q + self.logdet[c] + D * np.log(2 * np.pi)))
        return torch.logsumexp(torch.stack(lp), 0)


# ----------------------------------------------------------------------------- SMC sampler
def next_delta(L, beta, ess_frac):
    """Largest temperature increment keeping ESS >= ess_frac * N (per run, bisection)."""
    N = L.shape[1]
    Lc = L - L.max(1, keepdim=True).values

    def ess(d):
        w = torch.exp(d[:, None] * Lc)
        return w.sum(1) ** 2 / (w**2).sum(1)

    full = 1.0 - beta
    lo, hi = torch.zeros_like(beta), full.clone()
    for _ in range(80):
        mid = (lo + hi) / 2
        good = ess(mid) >= ess_frac * N
        lo, hi = torch.where(good, mid, lo), torch.where(good, hi, mid)
    d = torch.where(lo > 0, lo, hi)
    return torch.where(ess(full) >= ess_frac * N, full, d)


def systematic(lw, gen):
    r, N = lw.shape
    c = torch.softmax(lw, 1).cumsum(1)
    c[:, -1] = 1.0
    u = (torch.rand(r, 1, generator=gen, dtype=DT) + torch.arange(N, dtype=DT)) / N
    return torch.searchsorted(c, u).clamp(max=N - 1)


def mcmc(like, X, L, beta, runs, scale, steps, gen, target=0.25):
    """Differential-evolution Metropolis moves (ter Braak 2006): x + g (x_a - x_b) + small jitter
    along another particle difference, with a, b, c, d drawn from the whole population. Pairs from
    the same mode give proposals shaped like that mode however sloppy it is; pairs from different
    modes give long jumps that are mostly rejected (and occasionally move a particle between modes
    of similar shape, with g = 1). The step multiplier adapts towards ~25% acceptance."""
    r, N, _ = X.shape
    acc_rate = torch.zeros(r, dtype=DT)
    ar = torch.arange(r)[:, None]
    for _ in range(steps):
        ia, ib, ic, idd = (torch.randint(0, N, (r, N), generator=gen) for _ in range(4))
        diff = X[ar, ia] - X[ar, ib]
        jit = X[ar, ic] - X[ar, idd]
        g = torch.where(torch.rand(r, N, 1, generator=gen, dtype=DT) < 0.1, torch.ones(1, dtype=DT),
                        scale[:, None, None] * 2.38 / np.sqrt(2 * D))
        Xp = X + g * diff + 0.01 * torch.randn(r, N, 1, generator=gen, dtype=DT) * jit
        inside = ((Xp >= 0) & (Xp <= 1)).all(-1)
        Lp = like(Xp.clamp(0, 1), runs)
        Lp = torch.where(inside, Lp, torch.full_like(Lp, -torch.inf))
        acc = torch.log(torch.rand(r, N, generator=gen, dtype=DT)) < beta[:, None] * (Lp - L)
        X = torch.where(acc[..., None], Xp, X)
        L = torch.where(acc, Lp, L)
        acc_rate = acc.to(DT).mean(1)
        scale = (scale * torch.exp(acc_rate - target)).clamp(1e-2, 2.0)
    return X, L, scale, acc_rate


CKPT_BETAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def smc(like, R, N, seed, ess_frac=0.5, n_mcmc=10, n_final=30, max_stages=400, n_ckpt=4, verbose=False):
    """Tempered SMC for R runs at once. Also keeps n_ckpt random particles from the first stage at
    which beta passes each of CKPT_BETAS: these intermediate-temperature particles spread over every
    basin before tempering can empty a minority mode, and seed the mode census."""
    gen = torch.Generator().manual_seed(seed)
    nb = len(CKPT_BETAS)
    ckpt = torch.zeros(R, nb * n_ckpt, D, dtype=DT)
    have = torch.zeros(R, nb, dtype=torch.bool)
    X = torch.rand(R, N, D, generator=gen, dtype=DT)
    L = like(X, torch.arange(R))
    beta = torch.zeros(R, dtype=DT)
    logZ = torch.zeros(R, dtype=DT)
    scale = torch.ones(R, dtype=DT)
    n_st = torch.zeros(R, dtype=torch.long)
    t0 = time.time()
    for stage in range(max_stages):
        act = torch.nonzero(beta < 1).squeeze(1)
        if len(act) == 0:
            break
        Xa, La, ba = X[act], L[act], beta[act]
        d = next_delta(La, ba, ess_frac)
        lw = d[:, None] * La
        logZ[act] += torch.logsumexp(lw, 1) - np.log(N)
        idx = systematic(lw, gen)
        Xa = torch.gather(Xa, 1, idx[..., None].expand(-1, -1, D))
        La = torch.gather(La, 1, idx)
        ba = torch.where(ba + d >= 1 - 1e-12, torch.ones_like(ba), ba + d)
        Xa, La, sc, _ = mcmc(like, Xa, La, ba, act, scale[act], n_mcmc, gen)
        X[act], L[act], beta[act], scale[act] = Xa, La, ba, sc
        n_st[act] += 1
        for j, b in enumerate(CKPT_BETAS):
            new = (ba >= b) & ~have[act, j]
            for q in torch.nonzero(new).squeeze(1).tolist():
                sel = torch.randint(0, N, (n_ckpt,), generator=gen)
                ckpt[act[q], j * n_ckpt:(j + 1) * n_ckpt] = Xa[q, sel]
                have[act[q], j] = True
        if verbose and stage % 10 == 0:
            print(f"  stage {stage}: {len(act)} runs active, median beta {ba.median():.2e} ({time.time() - t0:.0f}s)",
                  flush=True)
    X, L, scale, acc = mcmc(like, X, L, beta, torch.arange(R), scale, n_final, gen)
    for j in range(nb):  # thresholds never passed (unconverged runs): use final particles
        for r in torch.nonzero(~have[:, j]).squeeze(1).tolist():
            ckpt[r, j * n_ckpt:(j + 1) * n_ckpt] = X[r, :n_ckpt]
    return dict(X=X, L=L, logZ=logZ, n_stages=n_st, acc=acc, converged=(beta >= 1), ckpt=ckpt)


# ----------------------------------------------------------------------------- mode analysis
def greedy_groups(U, chi2, radius, max_groups):
    """Group minima within `radius` of a group's best member, best chi^2 first. Returns labels and representatives."""
    order = np.argsort(chi2)
    reps, labels = [], np.full(len(U), -1)
    for i in order:
        for g, r in enumerate(reps):
            if np.linalg.norm(U[i] - U[r]) < radius:
                labels[i] = g
                break
        else:
            if len(reps) < max_groups:
                reps.append(i)
                labels[i] = len(reps) - 1
            else:  # attach to the nearest existing group
                labels[i] = int(np.argmin([np.linalg.norm(U[i] - U[r]) for r in reps]))
    return labels, reps


def is_mass(like, runs, centres, covs, n_is, gen, nu=4.0):
    """Importance sampling around each centre with a multivariate-t proposal (nu dof, scale covs).
    Returns proposal points X [K, n_is, 9] and log weights L - log q (-inf outside the box)."""
    K = len(centres)
    Lc = torch.linalg.cholesky(covs)
    z = torch.randn(K, n_is, D, generator=gen, dtype=DT)
    w = torch.distributions.Chi2(torch.tensor(nu, dtype=DT)).sample((K, n_is, 1))
    X = centres[:, None] + (z @ Lc.transpose(1, 2)) * torch.sqrt(nu / w)
    m = (z**2).sum(-1) * nu / w[..., 0]
    logq = (torch.lgamma(torch.tensor((nu + D) / 2, dtype=DT)) - torch.lgamma(torch.tensor(nu / 2, dtype=DT))
            - D / 2 * np.log(nu * np.pi) - torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum(1)[:, None]
            - (nu + D) / 2 * torch.log1p(m / nu))
    inside = ((X >= 0) & (X <= 1)).all(-1)
    L = like(X.clamp(0, 1), runs)
    return X, torch.where(inside, L - logq, torch.full_like(L, -torch.inf))


def mode_masses(X, logw, centres, covs):
    """Log mass of each of one run's modes. A proposal point counts for its own mode only if it is
    nearer, in every mode's own metric, to that mode's centre than to the others' (so overlapping
    proposals do not count the same posterior mass twice). Also returns the IS effective sample size."""
    K, n, _ = X.shape
    Linv = torch.linalg.inv(torch.linalg.cholesky(covs))
    d = X[:, :, None, :] - centres[None, None]  # [K, n, K, 9]
    maha = (torch.einsum("cij,kncj->knci", Linv, d) ** 2).sum(-1)  # [K(proposal), n, K(centre)]
    lm, ess = [], []
    for k in range(K):
        own = maha[k].argmin(-1) == k
        lw = torch.where(own, logw[k], torch.full_like(logw[k], -torch.inf))
        lm.append(torch.logsumexp(lw, 0).item() - np.log(n))
        ess.append(float(torch.exp(2 * torch.logsumexp(lw, 0) - torch.logsumexp(2 * lw, 0))))
    return lm, ess


def find_modes(models, lo, hi, data_ppm, sigma_ppm, res, a, seed):
    """Mode census for runs that share sigma. res: SMC output (final particles X [R, N, 9] and
    intermediate-temperature particles ckpt [R, n, 9]). Returns per-run mode summaries."""
    X, ck = res["X"], res["ckpt"]
    R, N, _ = X.shape
    g = np.random.default_rng(seed)
    nd = a.n_descents
    pick = np.stack([g.choice(N, nd, replace=False) for _ in range(R)])
    starts = torch.cat([torch.stack([X[r, pick[r]] for r in range(R)]), ck], 1)  # [R, M, 9]
    M = starts.shape[1]
    data = np.repeat(data_ppm * 1e-6, M, axis=0)
    f = Chi2(models, lo, hi, data, sigma_ppm * 1e-6)
    U = starts.reshape(-1, D)
    for _ in range(a.restarts):
        U = descend(f, U, a.descent_steps)
    with torch.no_grad():
        c = f(U[:, None])[:, 0].numpy()
    U, c = U.numpy().reshape(R, M, D), c.reshape(R, M)
    groups, pairs = [], []
    for r in range(R):
        lab, reps = greedy_groups(U[r], c[r], a.group_radius, a.max_groups)
        groups.append((lab, reps))
        for k in range(1, len(reps)):
            pairs.append((r, reps[0], reps[k], k))
    out = [dict(n_groups=len(gr[1]), chi2_min=float(c[r, gr[1][0]]), u_best=U[r, gr[1][0]], barriers=[],
                weight_separate=0.0, weight_ambiguous=0.0, separate=[], sep_groups=[]) for r, gr in enumerate(groups)]
    if pairs:
        rr = np.array([p[0] for p in pairs])
        ua = torch.as_tensor(np.stack([U[p[0], p[1]] for p in pairs]))
        ub = torch.as_tensor(np.stack([U[p[0], p[2]] for p in pairs]))
        fp = Chi2(models, lo, hi, data_ppm[rr] * 1e-6, sigma_ppm * 1e-6)
        path, _ = string_method(fp, ua, ub, a.images, a.string_iters)
        with torch.no_grad():
            cp = fp(upsample(path, 4))
        bar = (cp.max(1).values - torch.maximum(cp[:, 0], cp[:, -1])).clamp_min(0).numpy()
        cand = np.flatnonzero(bar >= a.sep)
        if len(cand):
            ra = SimpleNamespace(n_routes=a.n_routes, images=a.images, iters=a.refine_iters, detour=0.3,
                                 seed=seed, restarts=2, descent_steps=a.descent_steps)
            ref = refine_group(models, lo, hi, data_ppm[rr[cand]] * 1e-6, ua[cand], ub[cand], sigma_ppm * 1e-6, ra)
            bar[cand] = ref["barrier_best"]
        for (r, ia, ib, k), b in zip(pairs, bar):
            lab = groups[r][0]
            wgt = float((lab[:nd] == k).mean())  # share of the final posterior samples
            out[r]["barriers"].append(float(b))
            if b >= a.sep:
                out[r]["weight_separate"] += wgt
                out[r]["sep_groups"].append(ib)
                out[r]["separate"].append({"smc_weight": wgt, "barrier": float(b),
                                           "delta_chi2": float(c[r, ib] - c[r, ia]), "u": U[r, ib].round(4).tolist()})
            elif b >= a.one_basin:
                out[r]["weight_ambiguous"] += wgt
    # mass of every separate mode by importance sampling around it; the main mode's mass is the SMC
    # evidence (tempering tracks the main basin well; a Gaussian-shaped proposal does not, as the
    # main posterior is a curved valley clipped by the box)
    cen, run_of = [], []
    for r, o in enumerate(out):
        for ib in o["sep_groups"]:
            cen.append(U[r, ib])
            run_of.append(r)
    run_of = np.array(run_of, dtype=int)
    lm_all, ess_all = np.zeros(len(cen)), np.zeros(len(cen))
    if cen:
        cen_np = np.stack(cen)
        _, Jm = derivatives(models, lo + cen_np * (hi - lo), lo, hi)
        J = Jm.mean(0)
        F = np.swapaxes(J, 1, 2) @ J / sigma_ppm**2 + PRIOR_PRECISION * np.eye(D)
        covs = torch.as_tensor(np.linalg.inv(F) * a.is_inflate**2)
        like = EmuLike(models, lo, hi, data_ppm, np.full(len(data_ppm), sigma_ppm))
        cen_t = torch.as_tensor(cen_np)
        Xis, logw = is_mass(like, torch.as_tensor(run_of), cen_t, covs, a.n_is, torch.Generator().manual_seed(seed))
        for r in np.unique(run_of):
            ks = np.flatnonzero(run_of == r)
            lm, ess = mode_masses(Xis[ks], logw[ks], cen_t[ks], covs[ks])
            lm_all[ks], ess_all[ks] = lm, ess
    logZ = res["logZ"].numpy()
    for r, o in enumerate(out):
        ks = np.flatnonzero(run_of == r)
        o["log_mass_separate"], o["is_ess"] = lm_all[ks].tolist(), ess_all[ks].tolist()
        frac_is = 0.0
        if len(ks):
            ls = np.logaddexp.reduce(lm_all[ks])
            frac_is = float(np.exp(ls - np.logaddexp(ls, logZ[r])))
        o["mass_fraction_separate"] = frac_is
        o["max_barrier"] = max(o["barriers"]) if o["barriers"] else 0.0
        o["multimodal_smc"] = o["weight_separate"] >= a.min_weight
        o["multimodal_is"] = frac_is >= a.min_weight
        o["multimodal"] = o["multimodal_smc"] or o["multimodal_is"]
        del o["sep_groups"]
    return out


# ----------------------------------------------------------------------------- geometry at the truth
def truth_geometry(models, theta0, lo, hi, eps_ppm, sigmas, n_mc=512, seed=0):
    """Shape operators at the true planets; focal statistics for the actual noise and over noise draws."""
    _, Jm, Hm = derivatives(models, theta0, lo, hi, hessian=True)
    J, H = Jm.mean(0), Hm.mean(0)
    lam, V, B = shape_operators(J, H)
    # parameter-effects (tangential) part of the second derivatives, in the same whitened frame
    Jt, Ht = torch.as_tensor(J), torch.as_tensor(H)
    Q = torch.linalg.qr(Jt).Q
    T = torch.einsum("nkt,nlt,nlab->nkab", Q, Q, Ht)
    W = V / lam.clamp_min(1e-300).sqrt()[:, None, :]
    BT = torch.einsum("nja,nkjl,nlb->nkab", W, T, W)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_mc, B.shape[1], generator=g, dtype=DT)
    out = {"lam_u": lam.numpy(), "V_u": V.numpy()}
    eps = torch.as_tensor(eps_ppm)  # [n, n_sigma, 52]
    for j, s in enumerate(sigmas):
        mask = lam > PRIOR_PRECISION * s**2
        Bm = B * mask[:, None, :, None] * mask[:, None, None, :]
        out[f"Lambda_full_{s:g}"] = torch.linalg.eigvalsh(torch.einsum("nk,nkab->nab", eps[:, j], B))[:, -1].numpy()
        out[f"Lambda_dc_{s:g}"] = torch.linalg.eigvalsh(torch.einsum("nk,nkab->nab", eps[:, j], Bm))[:, -1].numpy()
        lf = torch.linalg.eigvalsh(torch.einsum("mk,nkab->nmab", z, Bm))[..., -1]
        out[f"P_focal_dc_{s:g}"] = (lf > 1 / s).to(DT).mean(1).numpy()
        out[f"gamma_dc_{s:g}"] = (kappa_max(B, mask=mask)[0] * s * 3.0).numpy()
        out[f"gammaT_dc_{s:g}"] = (kappa_max(BT, mask=mask)[0] * s * 3.0).numpy()
        # Fisher + uniform-prior (Gaussian) prediction of each parameter's posterior width (unit box), as in step 3
        post = torch.einsum("njk,nk,nik->nji", V, 1 / (lam / s**2 + PRIOR_PRECISION), V)
        out[f"pred_std_{s:g}"] = torch.sqrt(torch.diagonal(post, dim1=1, dim2=2)).numpy()
        out[f"n_dc_{s:g}"] = mask.sum(1).numpy()
        out[f"N_eff_{s:g}"] = (lam / (lam + PRIOR_PRECISION * s**2)).sum(1).numpy()
    return out


# ----------------------------------------------------------------------------- main experiment
def select_planets(code, a):
    z = np.load(os.path.join(a.curvature_dir, f"{code}_curvature.npz"))
    idx = z["index"]
    P = (z["focal_lambda_dc_30"].astype(np.float64) > 1 / 30).mean(1)
    rel = z["ratio_kappa_dc_30"] < 1.5
    g = np.random.default_rng(a.seed)
    n_r, n_e = (a.n_random, a.n_enriched) if code == a.primary else (a.n_random_check, a.n_enriched_check)
    rnd = g.choice(len(idx), n_r, replace=False)
    rest = np.setdiff1d(np.flatnonzero(rel), rnd)
    enr = rest[np.argsort(-P[rest])][:n_e]
    pos = np.r_[rnd, enr]
    return idx[pos], np.r_[np.zeros(n_r, int), np.ones(n_e, int)], P[pos]


def run_code(code, a):
    t0 = time.time()
    models = frozen_ensemble(code, a.emulator_dir)
    lo, hi = box(a.data_dir)
    from emulator import load_data
    S, P, _, _ = load_data(code, a.data_dir)
    planets, group, P_sel = select_planets(code, a)
    sigmas = a.sigmas if code == a.primary else a.sigmas[:1]
    theta0 = P[planets]
    with torch.no_grad():
        S0 = torch.stack([m(torch.as_tensor(theta0)) for m in models]).mean(0).numpy() * 1e6
    g = np.random.default_rng([a.seed, CODES.index(code)])
    eps = np.stack([g.normal(scale=s, size=S0.shape) for s in sigmas], 1)  # [n, n_sigma, 52]
    geo = truth_geometry(models, theta0, lo, hi, eps, sigmas, seed=a.seed)
    print(f"[{code}] geometry at {len(planets)} planets ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for j, s in enumerate(sigmas):
        data = S0 + eps[:, j]
        like = EmuLike(models, lo, hi, data, np.full(len(data), s))
        res = smc(like, len(data), a.n_particles, seed=a.seed + j, n_mcmc=a.n_mcmc, verbose=True)
        print(f"[{code} σ={s:g}] SMC: {like.n_eval / 1e6:.1f}M evaluations, stages median "
              f"{res['n_stages'].median().item()} ({time.time() - t0:.0f}s)", flush=True)
        modes = find_modes(models, lo, hi, data, s, res, a, seed=a.seed + j)
        post_std = res["X"].std(1).numpy()  # [R, 9] actual marginal widths (unit box)
        wr = post_std / geo[f"pred_std_{s:g}"]
        print(f"[{code} σ={s:g}] modes and shapes ({time.time() - t0:.0f}s)", flush=True)
        for i in range(len(planets)):
            m = modes[i]
            rows.append({
                "code": code, "planet": int(planets[i]), "group": ["random", "enriched"][group[i]], "sigma": s,
                "feature_ppm": float(S0[i].max() - S0[i].min()),
                "logZ": float(res["logZ"][i]), "n_stages": int(res["n_stages"][i]), "acc_final": float(res["acc"][i]),
                "smc_converged": bool(res["converged"][i]),
                "chi2_min": m["chi2_min"], "n_groups": m["n_groups"], "max_barrier": m["max_barrier"],
                "weight_separate": m["weight_separate"], "weight_ambiguous": m["weight_ambiguous"],
                "multimodal": bool(m["multimodal"]), "multimodal_smc": bool(m["multimodal_smc"]),
                "multimodal_is": bool(m["multimodal_is"]), "mass_fraction_separate": m["mass_fraction_separate"],
                "log_mass_separate": m["log_mass_separate"], "is_ess": m["is_ess"], "separate_modes": m["separate"],
                "width_ratio": wr[i].round(4).tolist(),
                "width_error": float(np.median(np.abs(np.log(wr[i])))),
                "gammaT_dc": float(geo[f"gammaT_dc_{s:g}"][i]),
                "Lambda_full": float(geo[f"Lambda_full_{s:g}"][i]), "Lambda_dc": float(geo[f"Lambda_dc_{s:g}"][i]),
                "P_focal_dc": float(geo[f"P_focal_dc_{s:g}"][i]), "gamma_dc": float(geo[f"gamma_dc_{s:g}"][i]),
                "n_dc": int(geo[f"n_dc_{s:g}"][i]), "N_eff": float(geo[f"N_eff_{s:g}"][i]),
                "posterior_mean_u": res["X"][i].mean(0).numpy().round(5).tolist(),
                "posterior_std_u": res["X"][i].std(0).numpy().round(6).tolist(),
                "truth_u": ((theta0[i] - lo) / (hi - lo)).round(5).tolist(),
            })
        np.savez_compressed(os.path.join(a.out, f"{code}_sigma{s:g}_samples.npz"), planets=planets,
                            X=res["X"][:, :a.n_save].numpy().astype(np.float32), data_ppm=data)
    return rows


def summarise(rows, a):
    out = {}
    bw = 0.3445  # Bates-Watts threshold for p = 9, n = 52
    for code in sorted({r["code"] for r in rows}):
        for s in sorted({r["sigma"] for r in rows if r["code"] == code}):
            R = [r for r in rows if r["code"] == code and r["sigma"] == s]
            e = {"n_runs": len(R), "smc_converged": float(np.mean([r["smc_converged"] for r in R])),
                 "smc_stages_median": float(np.median([r["n_stages"] for r in R]))}
            for grp in ("random", "enriched"):
                G = [r for r in R if r["group"] == grp]
                if not G:
                    continue
                mm = np.array([r["multimodal"] for r in G])
                mm_smc = np.array([r["multimodal_smc"] for r in G])
                mm_is = np.array([r["multimodal_is"] for r in G])
                ld = np.array([r["Lambda_dc"] > 1 for r in G])
                lf = np.array([r["Lambda_full"] > 1 for r in G])
                e[grp] = {
                    "n": len(G), "frac_multimodal": float(mm.mean()), "frac_multimodal_smc": float(mm_smc.mean()),
                    "frac_multimodal_is": float(mm_is.mean()),
                    "frac_with_separate_mode_any_weight": float(np.mean([len(r["separate_modes"]) > 0 for r in G])),
                    "frac_any_ambiguous": float(np.mean([r["weight_ambiguous"] > 0 for r in G])),
                    "frac_multiple_groups": float(np.mean([r["n_groups"] > 1 for r in G])),
                    "mean_P_focal_dc_predicted": float(np.mean([r["P_focal_dc"] for r in G])),
                    "frac_noise_crossed_dc": float(ld.mean()), "frac_noise_crossed_all": float(lf.mean()),
                    "multimodal_given_crossed_dc": [int((mm & ld).sum()), int(ld.sum())],
                    "multimodal_given_not_crossed_dc": [int((mm & ~ld).sum()), int((~ld).sum())],
                    "frac_bates_watts_nonlinear": float(np.mean([r["gamma_dc"] > bw for r in G])),
                }
            gN = np.array([r["gamma_dc"] for r in R])
            gT = np.array([r["gammaT_dc"] for r in R])
            we = np.array([r["width_error"] for r in R])
            WR = np.array([r["width_ratio"] for r in R])
            e["fisher_width_test"] = {
                "median_width_ratio_per_param": dict(zip(PARAM_NAMES, np.median(WR, 0).round(3).tolist())),
                "frac_off_by_1.5x_per_param": dict(zip(PARAM_NAMES, (np.abs(np.log(WR)) > np.log(1.5)).mean(0).round(3).tolist())),
                "median_run_width_error_dex": float(np.median(we) / np.log(10)),
                "median_gammaN": float(np.median(gN)), "median_gammaT": float(np.median(gT)),
                "frac_gammaN_above_bw": float((gN > bw).mean()), "frac_gammaT_above_bw": float((gT > bw).mean()),
                "spearman_width_error_vs_gammaN": float(spearmanr(gN, we)[0]),
                "spearman_width_error_vs_gammaT": float(spearmanr(gT, we)[0]),
                "spearman_width_error_vs_feature": float(spearmanr([r["feature_ppm"] for r in R], we)[0]),
                "median_width_error_dex_gammaT_below_bw": float(np.median(we[gT <= bw]) / np.log(10)) if (gT <= bw).any() else None,
                "median_width_error_dex_gammaT_above_bw": float(np.median(we[gT > bw]) / np.log(10)) if (gT > bw).any() else None,
            }
            out[f"{code}_sigma{s:g}"] = e
    return out


def plots(rows, a):
    bw = 0.3445
    codes = sorted({r["code"] for r in rows})
    # 1. multimodality vs focal statistic of the actual noise draw
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for c in codes:
        for s, mk in zip(sorted({r["sigma"] for r in rows}), ("o", "s")):
            R = [r for r in rows if r["code"] == c and r["sigma"] == s]
            if not R:
                continue
            x = np.array([r["Lambda_dc"] for r in R])
            y = np.array([max(r["weight_separate"], r["mass_fraction_separate"]) for r in R])
            mm = np.array([r["multimodal"] for r in R])
            axs[0].scatter(np.clip(x, 1e-3, None), np.clip(y, 3e-3, None), s=np.where(mm, 30, 10), marker=mk,
                           facecolors=np.where(mm, CODE_COLOR[c], "none"), edgecolors=CODE_COLOR[c], alpha=0.7,
                           label=f"{CODE_LABEL[c]}, σ = {s:g}")
    axs[0].axvline(1, color="k", lw=0.8, ls="--", label="noise crossed the focal set (Λ = 1)")
    axs[0].axhline(a.min_weight, color="grey", lw=0.8, label=f"multimodal: separated weight ≥ {a.min_weight:g}")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Λ_dc = λ_max(Σ ε_k B_k) for the actual noise (data-constrained directions)")
    axs[0].set_ylabel("posterior weight of separate modes\n(max of SMC share and IS mass; floored)")
    axs[0].legend(fontsize=6.5)
    for ax, key, lab in ((axs[1], "gamma_dc", "intrinsic γ_N"), (axs[2], "gammaT_dc", "parameter-effects γ_T")):
        for c in codes:
            R = [r for r in rows if r["code"] == c]
            ax.scatter([max(r[key], 1e-4) for r in R], [np.exp(r["width_error"]) for r in R], s=8,
                       color=CODE_COLOR[c], alpha=0.5, label=CODE_LABEL[c])
        ax.axvline(bw, color="k", lw=0.8, ls="--", label="Bates–Watts limit")
        ax.axhline(1.5, color="grey", lw=0.8, label="widths off by 1.5×")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Bates–Watts {lab} at the truth (data-constrained)")
        ax.set_ylabel("median factor between posterior and\nFisher+prior widths (over 9 parameters)")
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "posterior_vs_geometry.png"), dpi=150)
    plt.close(fig)

    # 2. multimodal fraction by group and sigma
    fig, ax = plt.subplots(figsize=(7, 3.8))
    keys = sorted({(r["code"], r["sigma"]) for r in rows})
    for i, (c, s) in enumerate(keys):
        for j, grp in enumerate(("random", "enriched")):
            G = [r for r in rows if r["code"] == c and r["sigma"] == s and r["group"] == grp]
            if not G:
                continue
            f = np.mean([r["multimodal"] for r in G])
            pred = np.mean([r["Lambda_dc"] > 1 for r in G])
            x = i + (j - 0.5) * 0.35
            ax.bar(x, f, 0.33, color=CODE_COLOR[c], alpha=1 if grp == "random" else 0.5, hatch="" if grp == "random" else "//")
            ax.plot(x, pred, "kx")
            ax.text(x, f + 0.005, f"{len(G)}", ha="center", fontsize=6)
    ax.set_xticks(range(len(keys)), [f"{CODE_LABEL[c]}\nσ = {s:g}" for c, s in keys], fontsize=8)
    ax.set_ylabel("fraction of retrievals")
    ax.set_title("Bars: multimodal posteriors (solid: random planets, hatched: highest P_focal).\n"
                 "×: noise crossed the data-constrained focal set", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "multimodal_fraction.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- validation
def validate(a):
    out = {}
    # 1. toy: two sloppy Gaussians, weights 0.7/0.3, log Z = 0
    toy = ToyLike(4)
    t0 = time.time()
    res = smc(toy, 8, a.n_particles, seed=1, n_mcmc=a.n_mcmc)
    X = res["X"]
    d = torch.stack([(X - toy.mu[c]).norm(dim=-1) for c in range(2)])
    w1 = (d[0] < d[1]).to(DT).mean(1)
    out["toy"] = {"logZ": res["logZ"].tolist(), "weight_mode1": w1.tolist(), "true_weight_mode1": toy.w,
                  "n_stages": res["n_stages"].tolist(), "seconds": time.time() - t0}
    print("toy", json.dumps(out["toy"]), flush=True)
    # 2. dynesty vs SMC on a few planets
    import dynesty
    models = frozen_ensemble(a.primary, a.emulator_dir)
    lo, hi = box(a.data_dir)
    from emulator import load_data
    _, P, _, _ = load_data(a.primary, a.data_dir)
    planets, _, _ = select_planets(a.primary, a)
    pick = np.r_[planets[:2], planets[-2:]]  # two random, two with the highest P_focal
    theta0 = P[pick]
    with torch.no_grad():
        S0 = torch.stack([m(torch.as_tensor(theta0)) for m in models]).mean(0).numpy() * 1e6
    s = a.sigmas[0]
    data = S0 + np.random.default_rng(7).normal(scale=s, size=S0.shape)
    like = EmuLike(models, lo, hi, data, np.full(len(data), s))
    res = smc(like, len(data), a.n_particles, seed=2, n_mcmc=a.n_mcmc)
    rows = []
    for i in range(len(pick)):
        def ll(u, i=i):
            return float(like(torch.as_tensor(u, dtype=DT)[None, None], torch.tensor([i]))[0, 0])
        t1 = time.time()
        ns = dynesty.NestedSampler(ll, lambda u: u, D, nlive=a.dynesty_nlive, bound="multi", sample="rslice",
                                   rstate=np.random.default_rng(i))
        ns.run_nested(dlogz=0.1, print_progress=False)
        r = ns.results
        xs = r.samples_equal(rstate=np.random.default_rng(0))
        xm = res["X"][i].numpy()
        qd = np.percentile(xs, [16, 50, 84], 0)
        qs = np.percentile(xm, [16, 50, 84], 0)
        scale = (qd[2] - qd[0]) / 2
        rows.append({"planet": int(pick[i]), "logZ_dynesty": float(r.logz[-1]), "logZ_err_dynesty": float(r.logzerr[-1]),
                     "logZ_smc": float(res["logZ"][i]),
                     "max_abs_quantile_shift_in_sd": float(np.abs((qs - qd) / scale).max()),
                     "median_abs_quantile_shift_in_sd": float(np.median(np.abs((qs - qd) / scale))),
                     "sd_ratio_smc_over_dynesty": ((qs[2] - qs[0]) / (qd[2] - qd[0])).round(3).tolist(),
                     "dynesty_calls": int(sum(r.ncall)), "dynesty_seconds": time.time() - t1})
        print("dynesty", json.dumps(rows[-1]), flush=True)
    out["dynesty_vs_smc"] = rows
    with open(os.path.join(a.out, "validation.json"), "w") as f:
        json.dump(out, f, indent=1)


def controls(a):
    """Positive and negative controls for the whole pipeline. Step 4 built noise-free data placed at
    t x rho_c along a planet's most curved data-constrained normal; for t > 1 the chi^2 minimum
    splits in two (a pitchfork), for t < 1 it does not. Cases whose split has a barrier >= 2 at
    30 ppm are positive controls; the same planets at t = 0.5 are negative controls."""
    import json as _json
    rows_out = []
    for code in a.codes:
        with open(os.path.join(a.curvature_dir, "summary.json")) as f:
            val = _json.load(f)[code]["validation"]
        z = np.load(os.path.join(a.curvature_dir, f"{code}_curvature.npz"))
        pos_rows = [v for v in val if v["kind"].startswith("data") and v["t"] >= 1.5 and v["delta_chi2_at_30ppm"] >= 2]
        planets = sorted({v["point"] for v in pos_rows})
        cases = [(v["point"], v["t"], v["rho_c_ppm"], "positive") for v in pos_rows]
        cases += [(q, 0.5, next(v["rho_c_ppm"] for v in pos_rows if v["point"] == q), "negative") for q in planets]
        if not cases:
            continue
        models = frozen_ensemble(code, a.emulator_dir)
        lo, hi = box(a.data_dir)
        theta0 = z["theta"][[c[0] for c in cases]]
        with torch.no_grad():
            S0 = torch.stack([m(torch.as_tensor(theta0)) for m in models]).mean(0).numpy() * 1e6
        key = f"nstar_dc_{a.sigmas[0]:g}"
        data = np.stack([S0[i] + t * rho * z[key][q] for i, (q, t, rho, _) in enumerate(cases)])
        s = a.sigmas[0]
        like = EmuLike(models, lo, hi, data, np.full(len(data), s))
        res = smc(like, len(data), a.n_particles, seed=11, n_mcmc=a.n_mcmc)
        modes = find_modes(models, lo, hi, data, s, res, a, seed=11)
        for (q, t, rho, kind), m in zip(cases, modes):
            rows_out.append({"code": code, "point": int(q), "t": t, "rho_c_ppm": rho, "control": kind,
                             "multimodal": bool(m["multimodal"]), "multimodal_smc": bool(m["multimodal_smc"]),
                             "multimodal_is": bool(m["multimodal_is"]), "smc_weight_separate": m["weight_separate"],
                             "mass_fraction_separate": m["mass_fraction_separate"], "max_barrier": m["max_barrier"],
                             "separate_modes": m["separate"], "is_ess": m["is_ess"]})
            print("control", _json.dumps({k: v for k, v in rows_out[-1].items() if k != "separate_modes"}), flush=True)
    with open(os.path.join(a.out, "controls.json"), "w") as f:
        _json.dump(rows_out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", nargs="+", default=["prt", "pos", "taurex"])
    ap.add_argument("--primary", default="prt", help="code with the full sample and both noise levels")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[30, 50])
    ap.add_argument("--n-random", type=int, default=80)
    ap.add_argument("--n-enriched", type=int, default=80)
    ap.add_argument("--n-random-check", type=int, default=30)
    ap.add_argument("--n-enriched-check", type=int, default=30)
    ap.add_argument("--n-particles", type=int, default=800)
    ap.add_argument("--n-mcmc", type=int, default=8)
    ap.add_argument("--n-save", type=int, default=400)
    ap.add_argument("--n-descents", type=int, default=16)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--descent-steps", type=int, default=250)
    ap.add_argument("--group-radius", type=float, default=0.03)
    ap.add_argument("--max-groups", type=int, default=6)
    ap.add_argument("--images", type=int, default=16)
    ap.add_argument("--string-iters", type=int, default=200)
    ap.add_argument("--refine-iters", type=int, default=300)
    ap.add_argument("--n-routes", type=int, default=12)
    ap.add_argument("--one-basin", type=float, default=0.5)
    ap.add_argument("--sep", type=float, default=2.0)
    ap.add_argument("--min-weight", type=float, default=0.05)
    ap.add_argument("--n-is", type=int, default=1000, help="importance samples per mode")
    ap.add_argument("--is-inflate", type=float, default=1.5, help="scale of the IS proposal relative to the Laplace width")
    ap.add_argument("--dynesty-nlive", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--controls", action="store_true", help="positive/negative controls from the step-4 focal validation")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--emulator-dir", default="runs/emulator")
    ap.add_argument("--curvature-dir", default="runs/curvature")
    ap.add_argument("--out", default="runs/posterior")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.validate:
        validate(a)
        return
    if a.controls:
        controls(a)
        return
    rows_path = os.path.join(a.out, "runs.json")
    if a.plots_only:
        with open(rows_path) as f:
            rows = json.load(f)
    else:
        rows = []
        for c in a.codes:
            rows += run_code(c, a)
            with open(rows_path, "w") as f:
                json.dump(rows, f, indent=1)
    summ = summarise(rows, a)
    summ["settings"] = vars(a)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    plots(rows, a)


if __name__ == "__main__":
    main()
