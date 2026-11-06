"""Step 5 check: are the step-4 focal-split minima separate modes once all 9 parameters can move?

curvature.py --validate places noise-free data at t x rho_c along a planet's most curved
data-constrained normal and descends from the true planet a small step either way along the most
curved tangent. Its "delta_chi2_at_30ppm" is the chi^2 drop from the true planet (now a saddle) to
the better of the two minima: a barrier along that one line only. Here the two minima are
re-descended and joined by the string method with 12 routes (refine_modes.refine_group), which
gives the barrier in the full 9-D space, at 30 ppm, for the 8 cases posterior.py --controls uses.

    python check_split_barriers.py      # ~5 min; writes runs/posterior/control_split_barriers.json
"""
import json
from types import SimpleNamespace

import numpy as np
import torch

from geometry import box, frozen_ensemble, derivatives, eigh_desc
from mode_paths import Chi2, descend
from refine_modes import refine_group
cases = [("prt", 713, 1.5), ("prt", 713, 2.5), ("pos", 70, 2.5), ("pos", 454, 1.5), ("pos", 454, 2.5),
         ("taurex", 1678, 2.5), ("taurex", 1147, 1.5), ("taurex", 1147, 2.5)]
val = json.load(open("runs/curvature/summary.json"))
lo, hi = box()
out = []
for code in ["prt", "pos", "taurex"]:
    models = frozen_ensemble(code)
    z = np.load(f"runs/curvature/{code}_curvature.npz")
    for c, q, t in [x for x in cases if x[0] == code]:
        v = next(v for v in val[code]["validation"] if v["kind"].startswith("data") and v["point"] == q and v["t"] == t)
        th = z["theta"][q:q + 1]
        S, J = derivatives(models, th, lo, hi)
        S0, J = S.mean(0)[0], J.mean(0)[0]
        lam, V = eigh_desc(J.T @ J)
        W = V / np.sqrt(lam)[None]
        vs = W @ z["ustar_dc_30"][q]; vs /= np.linalg.norm(vs)
        data = (S0 + t * v["rho_c_ppm"] * z["nstar_dc_30"][q])[None] * 1e-6
        u0 = torch.as_tensor((th - lo) / (hi - lo))
        f = Chi2(models, lo, hi, data, 1e-6)
        up = descend(f, (u0 + 0.01 * torch.as_tensor(vs)).clamp(0, 1), 600)
        dn = descend(f, (u0 - 0.01 * torch.as_tensor(vs)).clamp(0, 1), 600)
        a = SimpleNamespace(n_routes=12, images=24, iters=400, detour=0.3, seed=1, restarts=2, descent_steps=300)
        r = refine_group(models, lo, hi, data, up, dn, 30e-6, a)
        row = dict(code=c, point=q, t=t, step4_delta_chi2_30=v["delta_chi2_at_30ppm"],
                   step4_split_distance=v["split_distance"], barrier_best_30=float(r["barrier_best"][0]),
                   barrier_straight_30=float(r["barrier_straight"][0]), chi2_a=float(r["chi2_a"][0]),
                   chi2_b=float(r["chi2_b"][0]), distance_after_redescent=float(r["minima_distance"][0]),
                   faces_a=int(((r["ua"][0] < 1e-6) | (r["ua"][0] > 1 - 1e-6)).sum()),
                   faces_b=int(((r["ub"][0] < 1e-6) | (r["ub"][0] > 1 - 1e-6)).sum()))
        print(json.dumps(row), flush=True)
        out.append(row)
json.dump(out, open("runs/posterior/control_split_barriers.json", "w"), indent=1)
