"""Step 2c: re-test every "separate and credible" mode-test result with a harder path search.

mode_paths.py calls two minima separate when the optimised path between them climbs at least
Δχ² = 2 above the worse minimum. That barrier is an upper bound: a straight-line start with 12
images can settle on a ridge crossing when a lower pass exists elsewhere, or run out of
iterations. Here each such case is redone with twice the images, more iterations, and several
starting routes (the straight line plus random detours through a displaced midpoint). The
lowest barrier over all routes is kept, because one low route is enough to join the minima.

The data are replayed exactly, including the noise draws in the noisy run, from the run's own
settings in summary.json and the same random seed. The replay is checked against the saved
reference indices.

    python refine_modes.py                          # both runs/mode_paths and runs/mode_paths_noisy
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from emulator import PARAM_NAMES
from mode_paths import (CHI2_95, CODES, Chi2, classify, credible, descend, iter_sigmas, prepare,
                        string_method, upsample)


def routes(ua, ub, n_routes, n_images, detour, gen):
    """[n_routes, n_images, 9]: route 0 is the straight line, the rest bend through a random midpoint."""
    t = torch.linspace(0, 1, n_images, dtype=ua.dtype).view(-1, 1)
    out = [ua * (1 - t) + ub * t]
    for _ in range(n_routes - 1):
        d = torch.randn(ua.shape, dtype=ua.dtype, generator=gen)
        m = ((ua + ub) / 2 + detour * d / d.norm()).clamp(0, 1)
        tt = (2 * t).clamp(max=1)
        first = ua * (1 - tt) + m * tt
        tt = (2 * t - 1).clamp(min=0)
        second = m * (1 - tt) + ub * tt
        out.append(torch.where(t <= 0.5, first, second))
    return torch.stack(out)


def refine_group(models, lo, hi, data, ua, ub, sigma, a):
    """data [K, 52], ua/ub [K, 9]: returns per-case best barrier over routes, plus diagnostics."""
    K, R = len(data), a.n_routes
    gen = torch.Generator().manual_seed(a.seed)
    # re-descend both endpoints with fresh damping: first-pass descents can stall with the
    # damping at its cap, leaving an endpoint on a slope rather than at a minimum
    f0 = Chi2(models, lo, hi, data, sigma)
    for _ in range(a.restarts):
        ua, ub = descend(f0, ua, a.descent_steps), descend(f0, ub, a.descent_steps)
    init = torch.cat([routes(ua[k], ub[k], R, a.images, a.detour, gen) for k in range(K)])
    f = Chi2(models, lo, hi, np.repeat(data, R, axis=0), sigma)
    ua_r, ub_r = init[:, 0], init[:, -1]
    path, hist = string_method(f, ua_r, ub_r, a.images, a.iters, init=init)
    fine = upsample(path, 4)
    with torch.no_grad():
        c = f(fine)
        cm = f(fine, members=True)
    ends = torch.maximum(c[:, 0], c[:, -1])
    bar = (c.max(1).values - ends).clamp_min(0).view(K, R)
    bar_m = (cm.max(-1).values - torch.maximum(cm[..., 0], cm[..., -1])).clamp_min(0)  # [n_m, K*R]
    late = hist[(3 * len(hist)) // 4] - ends
    conv = ((late - (c.max(1).values - ends)).abs() <= 0.1 * late.abs() + 0.5).view(K, R)
    best = bar.argmin(1)
    idx = torch.arange(K) * R + best
    dist = (path[idx, 0] - path[idx, -1]).norm(dim=-1)
    return {
        "barrier_best": bar.min(1).values.numpy(),
        "barrier_straight": bar[:, 0].numpy(),
        "barrier_routes": bar.numpy(),
        "best_route": best.numpy(),
        "best_converged": conv[torch.arange(K), best].numpy(),
        "barrier_best_members": bar_m[:, idx].T.numpy(),
        "chi2_a": c[idx, 0].numpy(), "chi2_b": c[idx, -1].numpy(),
        "minima_distance": dist.numpy(),
        "ua": path[idx, 0].numpy(), "ub": path[idx, -1].numpy(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs/mode_paths", "runs/mode_paths_noisy"])
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--n-routes", type=int, default=16)
    ap.add_argument("--detour", type=float, default=0.3, help="midpoint displacement of bent routes (unit box)")
    ap.add_argument("--descent-steps", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=3, help="fresh-damping descents of each endpoint")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    for run in a.runs:
        t0 = time.time()
        with open(os.path.join(run, "summary.json")) as fh:
            args = SimpleNamespace(**json.load(fh)["args"])
        rows = []
        for code in args.codes:
            prep = prepare(code, args)
            rng = np.random.default_rng([args.seed, CODES.index(code)])
            for sigma, _, _, _, sets in iter_sigmas(prep, args, rng):
                z = np.load(os.path.join(run, f"{code}_sigma{sigma:g}.npz"))
                cases = []
                for name, (ii, _, _, data) in sets.items():
                    r = {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(name + "_")}
                    assert np.array_equal(r["ref"], ii), f"replay mismatch: {run} {code} {sigma} {name}"
                    _, sep, _, _ = classify(r, args)
                    for q in np.flatnonzero(sep & credible(r)):
                        cases.append((name, q, ii[q], data[q], r["ua"][q], r["ub"][q], r["barrier"][q]))
                if not cases:
                    continue
                res = refine_group(prep["models"], prep["lo"], prep["hi"], np.stack([c[3] for c in cases]),
                                   torch.as_tensor(np.stack([c[4] for c in cases])),
                                   torch.as_tensor(np.stack([c[5] for c in cases])), sigma * 1e-6, a)
                for k, (name, q, ref, _, _, _, b0) in enumerate(cases):
                    du = np.abs(res["ua"][k] - res["ub"][k])
                    rb = dict(barrier=res["barrier_best"][k], chi2_a=res["chi2_a"][k], chi2_b=res["chi2_b"][k],
                              ua=res["ua"][k][None], ub=res["ub"][k][None],
                              barrier_members=res["barrier_best_members"][k][None])
                    rb = {key: np.atleast_1d(v) for key, v in rb.items()}
                    one, sep, robust, _ = classify(rb, args)
                    verdict = "separate (robust)" if robust[0] else "separate" if sep[0] else \
                        "one basin" if one[0] else "ambiguous"
                    rows.append({
                        "code": code, "sigma_ppm": sigma, "set": name, "index": int(q), "ref": int(ref),
                        "barrier_first_pass": float(b0),
                        "barrier_straight_refined": float(res["barrier_straight"][k]),
                        "barrier_best": float(res["barrier_best"][k]),
                        "barrier_best_members": res["barrier_best_members"][k].round(3).tolist(),
                        "best_route": int(res["best_route"][k]),
                        "best_converged": bool(res["best_converged"][k]),
                        "chi2_a": float(res["chi2_a"][k]), "chi2_b": float(res["chi2_b"][k]),
                        "credible": bool(abs(res["chi2_a"][k] - res["chi2_b"][k]) < CHI2_95),
                        "minima_distance": float(res["minima_distance"][k]),
                        "largest_du": {PARAM_NAMES[d]: round(float(du[d]), 3) for d in np.argsort(-du)[:4]},
                        "verdict": verdict,
                    })
                    w = rows[-1]
                    print(f"{run} {code} σ={sigma:g} {name:7s} ref={ref:6d} barrier {b0:9.2f} -> {w['barrier_best']:9.2f} "
                          f"(straight {w['barrier_straight_refined']:9.2f}, route {w['best_route']:2d}, "
                          f"conv {w['best_converged']}) χ² {w['chi2_a']:.1f}/{w['chi2_b']:.1f} d={w['minima_distance']:.2f} -> {verdict}", flush=True)
        verdicts = {}
        for w in rows:
            key = f"{w['set']}"
            verdicts.setdefault(key, {}).setdefault(w["verdict"], 0)
            verdicts[key][w["verdict"]] += 1
        with open(os.path.join(run, "refine.json"), "w") as fh:
            json.dump({"settings": vars(a), "verdict_counts": verdicts, "cases": rows}, fh, indent=1)
        print(f"{run}: {len(rows)} cases, {verdicts} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
