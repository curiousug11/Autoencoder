"""Step 2a: neural-network emulators of the forward map theta -> spectrum, one per code.

Target. With Rp in Jupiter radii and Rs in solar radii, every spectrum satisfies
sqrt(S) * Rs / Rp >= RJ/Rsun = 0.10276: Rp is the radius at the bottom of the
atmosphere. So write

    S(lambda) = (0.10276 * Rp / Rs)^2 * (1 + y(lambda))^2,

where y = z / Rp is the height of the effective transit radius above Rp in
units of Rp. y spans about 1e-5 to 6, so the network predicts log y per bin and
the known (Rp/Rs)^2 factor is applied exactly. This keeps the emulator's error
roughly proportional to the size of the spectral features, not to the
much larger absolute depth.

Inputs: the 9 parameters scaled to the unit prior box, plus log Rs, log Rp,
log Mp, log T and log(T Rp / Mp) (scale height over Rp at fixed mean
molecular weight), all standardised. Activations are SiLU, so the network is
smooth and its first and second derivatives (needed for the Fisher metric and
curvature) are well defined.

26 POS spectra have y < 0 somewhere (transit depth below the bare planet),
all very hot, low-mass, puffy planets. They are treated as POS failures and
excluded from POS training and evaluation.

    python emulator.py --code pos --seed 0          # one member
    python emulator.py --code pos --seeds 0 1 2 3   # an ensemble, sequentially
    python emulator.py --code pos --evaluate        # test accuracy + Jacobian agreement
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

RJ_OVER_RSUN = 0.10276
PARAM_NAMES = ["Rs", "Rp", "Mp", "T", "chem1", "chem2", "chem3", "chem4", "chem5"]
SPLIT_SEED = 0  # train/val/test split is shared by every code and ensemble member


def load_data(code, data_dir="data"):
    S = np.load(os.path.join(data_dir, f"{code}_spec_binned.npy")).astype(np.float64)
    P = np.load(os.path.join(data_dir, f"{code}_params_clean.npy")).astype(np.float64)
    y = np.sqrt(S) * P[:, [0]] / (RJ_OVER_RSUN * P[:, [1]]) - 1.0
    valid = (y > 0).all(1)
    return S, P, y, valid


def split_indices(n, seed=SPLIT_SEED, frac=(0.8, 0.1)):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return perm[:a], perm[a:b], perm[b:]


class Emulator(nn.Module):
    """theta (physical units, shape [..., 9]) -> log y (shape [..., 52]) and spectrum."""

    def __init__(self, lo, hi, feat_mean, feat_std, out_mean, out_std, width=384, depth=4, n_out=52):
        super().__init__()
        for name, v in dict(lo=lo, hi=hi, feat_mean=feat_mean, feat_std=feat_std,
                            out_mean=out_mean, out_std=out_std).items():
            self.register_buffer(name, torch.as_tensor(v, dtype=torch.float64))
        layers, d = [], 14
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers).double()

    @staticmethod
    def raw_features(theta):
        Rs, Rp, Mp, T = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3]
        return torch.stack([torch.log(Rs), torch.log(Rp), torch.log(Mp), torch.log(T),
                            torch.log(T * Rp / Mp)], -1)

    def features(self, theta):
        u = (theta - self.lo) / (self.hi - self.lo)
        f = (self.raw_features(theta) - self.feat_mean) / self.feat_std
        return torch.cat([u, f], -1)

    def log_y(self, theta):
        return self.net(self.features(theta)) * self.out_std + self.out_mean

    def forward(self, theta):
        """Transit depth (fraction) for physical parameters theta."""
        y = torch.exp(self.log_y(theta))
        k = RJ_OVER_RSUN * theta[..., 1:2] / theta[..., 0:1]
        return (k * (1.0 + y)) ** 2

    def config(self):
        return {"width": self.net[0].out_features, "depth": (len(self.net) - 1) // 2}


def build(P_train, logy_train, width, depth):
    lo, hi = P_train.min(0), P_train.max(0)
    rf = Emulator.raw_features(torch.as_tensor(P_train)).numpy()
    return Emulator(lo, hi, rf.mean(0), rf.std(0), logy_train.mean(0), logy_train.std(0), width, depth)


def save(model, path, extra=None):
    torch.save({"state": model.state_dict(), "config": model.config(), "extra": extra or {}}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["state"]
    m = Emulator(st["lo"], st["hi"], st["feat_mean"], st["feat_std"], st["out_mean"], st["out_std"],
                 **ck["config"])
    m.load_state_dict(st)
    m.eval()
    return m


def load_ensemble(code, root="runs/emulator"):
    d = os.path.join(root, code)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("seed") and f.endswith(".pt"))
    return [load(p) for p in paths]


@torch.no_grad()
def predict(models, P, batch=20000):
    """Spectra [n_models, n, 52] for physical parameters P (numpy)."""
    th = torch.as_tensor(P, dtype=torch.float64)
    out = [torch.cat([m(th[a:a + batch]) for a in range(0, len(th), batch)]) for m in models]
    return torch.stack(out).numpy()


def train_one(code, seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    S, P, y, valid = load_data(code, args.data_dir)
    tr, va, te = split_indices(len(S))
    tr, va, te = tr[valid[tr]], va[valid[va]], te[valid[te]]
    logy = np.log(y.clip(1e-12))
    model = build(P[tr], logy[tr], args.width, args.depth)

    Pt = torch.as_tensor(P, dtype=torch.float64)
    Lt = torch.as_tensor((logy - logy[tr].mean(0)) / logy[tr].std(0), dtype=torch.float64)
    Xall = model.features(Pt).detach()
    opt = torch.optim.AdamW(model.net.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05, final_div_factor=1e3)
    hist = []
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.as_tensor(np.random.permutation(tr))
        for a in range(0, len(perm), args.batch):
            idx = perm[a:a + args.batch]
            loss = ((model.net(Xall[idx]) - Lt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = ((model.net(Xall[va]) - Lt[va]) ** 2).mean().item()
                tl = loss.item()
            hist.append((ep, tl, vl))
            if vl < best:
                best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"[{code} s{seed}] epoch {ep:4d} train {tl:.3e} val {vl:.3e} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    model.eval()

    # accuracy in ppm on the held-out test set
    pred = predict([model], P[te])[0]
    err = (pred - S[te]) * 1e6
    feat = (S[te].max(1) - S[te].min(1)) * 1e6
    rms = np.sqrt((err**2).mean(1))
    rich = feat >= 210
    metrics = {
        "code": code, "seed": seed, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "n_excluded_invalid": int((~valid).sum()), "best_val_loss": best, "train_time_s": time.time() - t0,
        "test_rms_ppm_median": float(np.median(rms)), "test_rms_ppm_p90": float(np.percentile(rms, 90)),
        "test_rms_ppm_p99": float(np.percentile(rms, 99)),
        "test_rms_ppm_median_feature_rich": float(np.median(rms[rich])),
        "test_rms_ppm_p90_feature_rich": float(np.percentile(rms[rich], 90)),
        "test_rms_over_feature_median_feature_rich": float(np.median(rms[rich] / feat[rich])),
        "args": vars(args),
    }
    out = os.path.join(args.out, code)
    os.makedirs(out, exist_ok=True)
    save(model, os.path.join(out, f"seed{seed}.pt"), metrics)
    with open(os.path.join(out, f"seed{seed}_metrics.json"), "w") as f:
        json.dump(metrics | {"history": hist}, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=1))
    return metrics


def jacobian_unit(model, P):
    """d spectrum / d u for unit-box coordinates u, shape [n, 52, 9]."""
    th = torch.as_tensor(P, dtype=torch.float64)
    scale = model.hi - model.lo
    f = lambda t: model(t)
    J = torch.func.vmap(torch.func.jacrev(f))(th)
    return (J * scale).detach().numpy()


def evaluate_ensemble(code, args):
    """Test accuracy of each member and the ensemble mean, plus Jacobian agreement between members."""
    S, P, y, valid = load_data(code, args.data_dir)
    _, _, te = split_indices(len(S))
    te = te[valid[te]]
    models = load_ensemble(code, args.out)
    pred = predict(models, P[te])
    feat = (S[te].max(1) - S[te].min(1)) * 1e6
    rich = feat >= 210

    def stats(p):
        rms = np.sqrt((((p - S[te]) * 1e6) ** 2).mean(1))
        return {"median": float(np.median(rms)), "p90": float(np.percentile(rms, 90)),
                "p99": float(np.percentile(rms, 99)),
                "median_feature_rich": float(np.median(rms[rich])),
                "p90_feature_rich": float(np.percentile(rms[rich], 90)),
                "p99_feature_rich": float(np.percentile(rms[rich], 99))}

    out = {"code": code, "n_members": len(models), "n_test": int(len(te)),
           "test_rms_ppm_members": [stats(p) for p in pred],
           "test_rms_ppm_ensemble_mean": stats(pred.mean(0)),
           "member_spread_rms_ppm_median": float(np.median(np.sqrt(((pred.std(0) * 1e6) ** 2).mean(1))))}

    # Jacobian agreement on feature-rich test spectra (derivatives in ppm per unit prior range)
    rng = np.random.default_rng(1)
    k = rng.choice(np.flatnonzero(rich), min(args.n_jac, rich.sum()), replace=False)
    J = np.stack([jacobian_unit(m, P[te][k]) for m in models]) * 1e6  # [m, n, 52, 9]
    Jm = J.mean(0)
    rel_frob = np.linalg.norm(J - Jm, axis=(2, 3)) / np.linalg.norm(Jm, axis=(1, 2))
    col = np.linalg.norm(J - Jm, axis=2) / np.linalg.norm(Jm, axis=1)  # [m, n, 9]
    out["jacobian"] = {
        "n_points": int(len(k)),
        "rel_frobenius_disagreement_median": float(np.median(rel_frob)),
        "rel_frobenius_disagreement_p90": float(np.percentile(rel_frob, 90)),
        "rel_column_disagreement_median": {p: float(np.median(col[..., q])) for q, p in enumerate(PARAM_NAMES)},
        "column_norm_median_ppm_per_unit": {p: float(np.median(np.linalg.norm(Jm[..., q], axis=1)))
                                            for q, p in enumerate(PARAM_NAMES)},
    }
    with open(os.path.join(args.out, code, "ensemble_eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", required=True, choices=["pos", "prt", "taurex"])
    ap.add_argument("--evaluate", action="store_true", help="evaluate the trained ensemble instead of training")
    ap.add_argument("--n-jac", type=int, default=1000, help="test points for the Jacobian check")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/emulator")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.evaluate:
        evaluate_ensemble(args.code, args)
        return
    for s in args.seeds:
        train_one(args.code, s, args)


if __name__ == "__main__":
    main()
