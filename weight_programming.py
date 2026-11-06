"""Weight-space programming on the best noiseless autoencoder (no gradient descent, no training).

The trained weights are read directly and new weights are written by hand, in the sense of
https://rikiyat.github.io/blog/weight-space-programming/: no optimiser, no fitting of weights
to data, no search. Forward passes are used only to *measure* a hand-designed edit on the
validation split (a small, fixed budget of discrete choices), and the test split is used once at
the end for reporting.

Base model: runs/ae_wide_cosine seed 42 (52-256-128-64-32-16-8, mirrored; GELU; POS; split 42).

1. Read (`weight_report.json`, `weights.png`)
   - The output layer's weakest left-singular direction is almost exactly the all-ones vector:
     the network has written "outputs have zero mean" (z-normalised spectra) into its weights.
   - The first layer's rows have large components along the all-ones vector. Those components
     do nothing, because every input is zero-mean.
   - No hidden unit is dead or always linear, and every weight matrix is full rank, so there
     is nothing to prune.

2. Edit: exact output centering. W_out <- (I - 11^T/52) W_out, b_out <- b_out - mean(b_out).
   For a zero-mean target x, ||x - P y||^2 = ||x - y||^2 - ||(I - P) y||^2, so this cannot
   increase the error of any spectrum.

3. Program: residual-feedback encoder (8-d code, original decoder).
   The decoder D is kept. The encoder is rewritten so that it computes a better code for D:
       u_0 = x,   u_{k+1} = u_k + alpha (x - AE(u_k)),   code = E(u_K),   output = D(code)
   AE(u) is close to a projection onto the spectrum manifold. Feeding the residual back
   corrects the part of the error that lies along the manifold (a Richardson iteration on the
   trained encoder/decoder pair). The decoder still sees only 8 numbers.
   The whole thing is written as one plain Linear/GELU encoder: the trained layers are copied
   in, consecutive linear maps are fused, and x and u_k are carried exactly through every GELU
   layer by hand-written unit pairs, because GELU(t) - GELU(-t) = t.
   Self-check (optional, selected on validation): the encoder also decodes E(x) and E(u_K)
   internally, compares their L1 residuals against x with hand-written units
   (|t| ~ (GELU(ct) + GELU(-ct))/c, a GELU step, and a gate z0 + s (z2 - z0) written as
   GELU(d + M(s-1)) - GELU(-d + M(s-1))), and outputs the better code.
   alpha, K and the self-check are chosen from a 24-point grid on the validation split.

4. Construction: parallel ensemble network. Independently trained autoencoders (same split)
   are stacked block-diagonally into one wider network whose output layer averages theirs.
   Exact, but the code is 8 x n_models numbers, so it is a different compression rate.

Usage:
    python weight_programming.py                   # read, edit, program, construct, evaluate
    python weight_programming.py --alphas 0.5 --iters 2

Outputs go to runs/weight_programming/. The programmed autoencoder is saved as
runs/weight_programming/feedback/recipe.pt; load_programmed() rebuilds the network from it.
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

from data import DATA_DIR, prepare_splits
from model import ACTIVATIONS, Autoencoder, mlp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BEST = "runs/ae_wide_cosine/scheduler_cosine/seed_42/model.pt"
SEEDS = ["runs/weight_averaging/seeds/seed_43/model.pt", "runs/weight_averaging/seeds/seed_44/model.pt"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--best", default=BEST)
    p.add_argument("--seeds", nargs="*", default=SEEDS,
                   help="other trained autoencoders (same split) for the parallel-ensemble construction")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--iters", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--out", default="runs/weight_programming")
    return p.parse_args()


# ---------------------------------------------------------------- loading / evaluation

def load_model(path):
    ck = torch.load(path, weights_only=False)
    a = ck["args"]
    model = Autoencoder(ck["n_in"], tuple(a["hidden"]), a["latent"], ACTIVATIONS[a["activation"]])
    model.load_state_dict(ck["model_state"])
    return model.eval(), ck


def load_splits(ck):
    a = ck["args"]
    seed = a["seed"] if a.get("split_seed") is None else a["split_seed"]
    splits, wavelength, _ = prepare_splits(a.get("data_dir", DATA_DIR), a["min_feature_height_ppm"], seed=seed,
                                           dataset=a.get("dataset", "pos"), cut_datasets=a.get("cut_datasets"))
    return splits, wavelength


@torch.no_grad()
def evaluate(model, split):
    x = torch.from_numpy(split.spectra)
    err = (model(x) - x).numpy()
    per = (err ** 2).mean(1)
    ppm = np.sqrt(((err * split.std[:, None]) ** 2).mean(1)) * 1e6
    return {"mse": float(per.mean()), "median": float(np.median(per)), "p95": float(np.percentile(per, 95)),
            "p99": float(np.percentile(per, 99)), "rms_ppm_median": float(np.median(ppm)),
            "rms_ppm_p99": float(np.percentile(ppm, 99)), "per_spectrum": per, "per_bin": (err ** 2).mean(0)}


def linears(seq):
    return [m for m in seq if isinstance(m, nn.Linear)]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- 1. reading the weights

@torch.no_grad()
def read_weights(model, split, wavelength, out):
    ones = torch.ones(52) / np.sqrt(52)
    report = {"layers": {}}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            s = torch.linalg.svdvals(mod.weight)
            report["layers"][name] = {"shape": list(mod.weight.shape), "fro": float(mod.weight.norm()),
                                      "sv_max": float(s[0]), "sv_min": float(s[-1]),
                                      "sv_min_over_max": float(s[-1] / s[0])}
    w_out = model.decoder[-1].weight
    u, s, _ = torch.linalg.svd(w_out)
    report["output_layer"] = {
        "weakest_left_singular_value": float(s[-1]), "next_weakest": float(s[-2]), "strongest": float(s[0]),
        "overlap_of_weakest_direction_with_ones": float((u[:, -1] @ ones).abs()),
        "sum_of_output_bias": float(model.decoder[-1].bias.sum()),
    }
    w_in = model.encoder[0].weight
    along_ones = (w_in @ ones).pow(2).sum() / w_in.pow(2).sum()
    report["input_layer"] = {"fraction_of_weight_energy_along_ones": float(along_ones)}

    # activation regimes (a measurement on the validation spectra)
    h, regimes = torch.from_numpy(split.spectra), []
    for mod in list(model.encoder) + list(model.decoder):
        if isinstance(mod, nn.GELU):
            regimes.append({"units": h.shape[1], "always_off": int(((h < -3).float().mean(0) > 0.999).sum()),
                            "always_linear": int(((h > 3).float().mean(0) > 0.999).sum()),
                            "min_preactivation_std": float(h.std(0).min())})
        h = mod(h)
    report["gelu_layers"] = regimes
    y = model(torch.from_numpy(split.spectra))
    report["outputs"] = {"rms_of_spectrum_mean": float(y.mean(1).pow(2).mean().sqrt()),
                         "mean_of_spectrum_std": float(y.std(1, unbiased=False).mean())}
    (out / "weight_report.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            sv = torch.linalg.svdvals(mod.weight).numpy()
            axes[0].plot(np.arange(1, len(sv) + 1), sv / sv[0], lw=1, label=name)
    axes[0].set(xlabel="index", ylabel="singular value / largest", yscale="log", title="Singular spectra (all full rank)")
    axes[0].legend(fontsize=5, ncol=2)
    order = np.argsort(wavelength)
    axes[1].plot(wavelength[order], u[order, -1].numpy() * float(np.sign(u[:, -1].sum())), "o-", ms=3)
    axes[1].axhline(1 / np.sqrt(52), color="k", ls="--", lw=1, label="all-ones / √52")
    axes[1].set(xscale="log", xlabel="wavelength [µm]", ylim=(0, 0.3),
                title=f"Output layer: weakest direction (σ={s[-1]:.3f} vs max {s[0]:.2f})")
    axes[1].legend(fontsize=8)
    im = axes[2].imshow(w_out[order].numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set(xlabel="last hidden unit", ylabel="output bin (short → long λ)", title="Output layer weights")
    fig.colorbar(im, ax=axes[2])
    fig.tight_layout()
    fig.savefig(out / "weights.png", dpi=150)
    plt.close(fig)
    return report


# ---------------------------------------------------------------- 2. exact output centering

def centred(model):
    """Project the output layer onto zero-mean spectra; never increases any spectrum's error."""
    new = copy.deepcopy(model)
    last = new.decoder[-1]
    with torch.no_grad():
        last.weight -= last.weight.mean(0, keepdim=True)
        last.bias -= last.bias.mean()
    return new.eval()


# ---------------------------------------------------------------- 3. residual-feedback encoder

class Affine:
    """A vector expressed as M @ state + v, for the current layer's state."""

    def __init__(self, M, v):
        self.M, self.v = M, v

    def __add__(self, o):
        return Affine(self.M + o.M, self.v + o.v)

    def __rmul__(self, c):
        return Affine(c * self.M, c * self.v)

    def __sub__(self, o):
        return self + (-1.0) * o

    def map(self, W, b=None):
        return Affine(W @ self.M, W @ self.v + (0 if b is None else b))


def scalar_sum(a, scale=1.0):
    """Affine for scale * (sum of the entries of a)."""
    return Affine(scale * a.M.sum(0, keepdim=True), scale * a.v.sum(0, keepdim=True))


def broadcast(a, n):
    return Affine(a.M.expand(n, -1).clone(), a.v.expand(n).clone())


class EncoderWriter:
    """Writes a plain Linear/GELU stack layer by layer.

    Each written layer has some blocks of pre-activations (the trained layers' units, or
    hand-written units) plus, for every carried vector c, the pair (+c, -c). After the GELU,
    c is recovered exactly as GELU(c) - GELU(-c).
    """

    def __init__(self, n_in):
        self.layers = []  # (W, b) per Linear
        self.dim = n_in

    def gelu_layer(self, blocks, carries):
        rows = list(blocks) + [r for c in carries.values() for r in (c, -1.0 * c)]
        W = torch.cat([r.M for r in rows])
        b = torch.cat([r.v for r in rows])
        self.layers.append((W, b))
        self.dim = W.shape[0]
        eye = torch.eye(self.dim, dtype=torch.float64)
        zeros = lambda n: torch.zeros(n, dtype=torch.float64)  # noqa: E731
        out, i = [], 0
        for blk in blocks:  # post-activation values of each block
            n = blk.M.shape[0]
            out.append(Affine(eye[i:i + n], zeros(n)))
            i += n
        new_carries = {}
        for name, c in carries.items():
            n = c.M.shape[0]
            new_carries[name] = Affine(eye[i:i + n] - eye[i + n:i + 2 * n], zeros(n))
            i += 2 * n
        return out, new_carries

    def linear_layer(self, out):
        self.layers.append((out.M, out.v))
        self.dim = out.M.shape[0]

    def build(self):
        sizes = [self.layers[0][0].shape[1]] + [W.shape[0] for W, _ in self.layers]
        enc = mlp(sizes, nn.GELU)
        with torch.no_grad():
            for lin, (W, b) in zip(linears(enc), self.layers):
                lin.weight.copy_(W.float())
                lin.bias.copy_(b.float())
        return enc


# constants of the hand-written selection logic (see feedback_autoencoder)
ABS_GAIN = 1e5    # |t| ~ (GELU(c t) + GELU(-c t)) / c
STEP_WIDTH = 1e-4  # residual difference over which the gate goes from 0 to 1
STEP_GAIN = 10.0  # clip(g, 0, 1) ~ (GELU(c g) - GELU(c (g - 1))) / c
BIG = 20.0        # z0 + s (z2 - z0) via GELU(d + BIG (s - 1)) - GELU(-d + BIG (s - 1)); |d| << BIG


def feedback_autoencoder(model, alpha, iters, select=False):
    """Single Linear/GELU encoder computing the code E(u_K), with u_0 = x and
    u_{k+1} = u_k + alpha (x - AE(u_k)). The decoder is the model's own (unchanged) decoder.

    With select=True the encoder also decodes both E(x) and E(u_K) internally, compares their
    L1 residuals against x, and outputs the code with the smaller residual. The comparison
    and the gate are hand-written GELU units.
    """
    E = [(m.weight.double(), m.bias.double()) for m in linears(model.encoder)]
    D = [(m.weight.double(), m.bias.double()) for m in linears(model.decoder)]
    n = E[0][0].shape[1]
    w = EncoderWriter(n)
    x = Affine(torch.eye(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64))
    c = {"x": x, "u": x}
    extra = []  # hand-written blocks to add to the next layer
    for it in range(iters):  # one unrolled copy of the autoencoder per iteration
        (h, *ex), c = w.gelu_layer([c["u"].map(*E[0]), *extra], c)
        if extra:  # the |x - AE(x)| units of the previous pass: keep their sum r0
            c["r0"] = scalar_sum(ex[0], 1 / ABS_GAIN) + scalar_sum(ex[1], 1 / ABS_GAIN)
            extra = []
        for W, b in E[1:-1]:
            (h,), c = w.gelu_layer([h.map(W, b)], c)
        z = h.map(*E[-1])
        if select and it == 0:
            c["z0"] = z
        # linear bottleneck: fuse the encoder's last layer with the decoder's first layer
        (h,), c = w.gelu_layer([z.map(*D[0])], c)
        for W, b in D[1:-1]:
            (h,), c = w.gelu_layer([h.map(W, b)], c)
        x_hat = h.map(*D[-1])  # never materialised: folded into the next layer
        if select and it == 0:
            res = ABS_GAIN * (c["x"] - x_hat)
            extra = [res, -1.0 * res]
        c["u"] = c["u"] + alpha * (c["x"] - x_hat)
    u = c.pop("u")
    if not select:
        c = {}
    # final pass: the original encoder applied to u_K
    (h, *ex), c = w.gelu_layer([u.map(*E[0]), *extra], c)
    if extra:
        c["r0"] = scalar_sum(ex[0], 1 / ABS_GAIN) + scalar_sum(ex[1], 1 / ABS_GAIN)
    for W, b in E[1:-1]:
        (h,), c = w.gelu_layer([h.map(W, b)], c)
    z = h.map(*E[-1])
    if not select:
        w.linear_layer(z)  # the explicit 8-d code
        return ProgrammedAE(w.build(), model.decoder).eval()

    # decode z = E(u_K) inside the encoder and measure its L1 residual r2
    c["z2"] = z
    (h,), c = w.gelu_layer([z.map(*D[0])], c)
    for W, b in D[1:-1]:
        (h,), c = w.gelu_layer([h.map(W, b)], c)
    res = ABS_GAIN * (c.pop("x") - h.map(*D[-1]))
    (a, b), c = w.gelu_layer([res, -1.0 * res], c)
    r2 = scalar_sum(a, 1 / ABS_GAIN) + scalar_sum(b, 1 / ABS_GAIN)
    # gate s = 1 if r2 < r0 (feedback code is better), else 0
    g = (1 / STEP_WIDTH) * (c.pop("r0") - r2)
    one = Affine(torch.zeros_like(g.M), torch.ones_like(g.v))
    (p, q), c = w.gelu_layer([STEP_GAIN * g, STEP_GAIN * (g - one)], c)
    s = (1 / STEP_GAIN) * (p - q)
    # code = z0 + s (z2 - z0); exact when s is 0 or 1
    d = c["z2"] - c["z0"]
    k = d.M.shape[0]
    shift = BIG * broadcast(s - Affine(torch.zeros_like(s.M), torch.ones_like(s.v)), k)
    (gp, gm), c = w.gelu_layer([d + shift, -1.0 * d + shift], {"z0": c["z0"]})
    w.linear_layer(c["z0"] + gp - gm)  # the explicit 8-d code
    return ProgrammedAE(w.build(), model.decoder).eval()


class ProgrammedAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


@torch.no_grad()
def feedback_reference(model, x, alpha, iters, select=False):
    """The same computation written as a loop, to check the hand-written network against."""
    u = x
    for _ in range(iters):
        u = u + alpha * (x - model(u))
    z0, z2 = model.encoder(x), model.encoder(u)
    if not select:
        return model.decoder(z2)
    r0 = (x - model.decoder(z0)).abs().sum(1, keepdim=True)
    r2 = (x - model.decoder(z2)).abs().sum(1, keepdim=True)
    return model.decoder(torch.where(r2 < r0, z2, z0))


def save_programmed(path, meta):
    """The programmed network is fully determined by the base weights and the recipe, so only the
    recipe is stored (the dense network is ~15 MB, mostly zeros and copies of the base weights)."""
    torch.save({"meta": meta}, path)


def load_programmed(path):
    """Rebuild the programmed autoencoder (a ProgrammedAE with plain Linear/GELU stacks)."""
    meta = torch.load(path, weights_only=False)["meta"]
    base = centred(load_model(meta["base"])[0])
    return feedback_autoencoder(base, meta["alpha"], meta["iters"], meta["mode"] == "select"), meta


# ---------------------------------------------------------------- 4. parallel ensemble network

def parallel_ensemble(models):
    """One wider Autoencoder whose output is exactly the mean of the given models' outputs."""
    hid = [m.out_features for m in linears(models[0].encoder)][:-1]
    lat = linears(models[0].encoder)[-1].out_features
    k = len(models)
    big = Autoencoder(52, tuple(k * h for h in hid), k * lat).eval()
    with torch.no_grad():
        for big_lin, *lins in zip(linears(big.encoder) + linears(big.decoder),
                                  *[linears(m.encoder) + linears(m.decoder) for m in models]):
            big_lin.weight.zero_()
            r = c = 0
            first, last = lins[0] is linears(models[0].encoder)[0], lins[0] is linears(models[0].decoder)[-1]
            for lin in lins:
                o, i = lin.weight.shape
                if first:      # every block reads the same input
                    big_lin.weight[r:r + o, :] = lin.weight
                elif last:     # output = mean of the blocks' outputs
                    big_lin.weight[:, c:c + i] = lin.weight / k
                else:          # block diagonal
                    big_lin.weight[r:r + o, c:c + i] = lin.weight
                r, c = r + o, c + i
            big_lin.bias.copy_(sum(lin.bias for lin in lins) / k if last else torch.cat([lin.bias for lin in lins]))
    return big


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, ck = load_model(args.best)
    splits, wavelength = load_splits(ck)
    x_val = torch.from_numpy(splits["val"].spectra)
    val = lambda m: evaluate(m, splits["val"])["mse"]  # noqa: E731

    report = read_weights(model, splits["val"], wavelength, out)
    print(json.dumps({k: report[k] for k in ("output_layer", "input_layer", "outputs")}, indent=2))

    candidates = {"original": model}
    base = centred(model)
    candidates["edit: output centering"] = base

    # choose alpha, K and selection on validation; each grid point is one measured configuration
    grid = {}
    with torch.no_grad():
        for select in (False, True):
            for alpha in args.alphas:
                for k in args.iters:
                    y = feedback_reference(base, x_val, alpha, k, select)
                    grid[f"{alpha}/{k}/{'select' if select else 'plain'}"] = float(((y - x_val) ** 2).mean())
    print("validation grid (alpha/K/mode -> val MSE):", {k: f"{v:.3e}" for k, v in grid.items()})
    chosen = {}
    for mode in ("plain", "select"):
        key = min((k for k in grid if k.endswith(mode)), key=grid.get)
        alpha, iters, _ = key.split("/")
        alpha, iters, select = float(alpha), int(iters), mode == "select"
        net = feedback_autoencoder(base, alpha, iters, select)
        with torch.no_grad():
            ref = feedback_reference(base, x_val, alpha, iters, select)
            diff = (net(x_val) - ref).abs().max(1).values
            assert float(((net(x_val) - ref) ** 2).mean()) < 1e-8
        print(f"{mode}: hand-written network vs loop: max |difference| {float(diff.max()):.2e}, "
              f"spectra differing by > 1e-4: {int((diff > 1e-4).sum())} (near-ties of the gate)")
        label = (f"program: residual feedback (alpha={alpha}, K={iters})" if not select else
                 f"program: feedback + self-check (alpha={alpha}, K={iters})")
        candidates[label] = net
        chosen[mode] = {"alpha": alpha, "iters": iters, "label": label, "net": net}
    best_mode = min(chosen, key=lambda mo: val(chosen[mo]["net"]))
    name_fb = chosen[best_mode]["label"]
    alpha, iters = chosen[best_mode]["alpha"], chosen[best_mode]["iters"]

    ensemble_models = [model] + [load_model(p)[0] for p in args.seeds]
    for p in args.seeds:
        a = torch.load(p, weights_only=False)["args"]
        assert (a["seed"] if a.get("split_seed") is None else a["split_seed"]) == 42, f"{p}: different split"
    if len(ensemble_models) > 1:
        ens = parallel_ensemble(ensemble_models)
        with torch.no_grad():
            ref = sum(m(x_val) for m in ensemble_models) / len(ensemble_models)
            assert float((ens(x_val) - ref).abs().max()) < 1e-4
        candidates[f"construction: parallel ensemble of {len(ensemble_models)} AEs"] = ens

    rows, per = {}, {}
    for name, m in candidates.items():
        code = linears(m.encoder)[-1].out_features
        t = evaluate(m, splits["test"])
        per[name] = t
        rows[name] = {"train_mse": evaluate(m, splits["train"])["mse"], "val_mse": val(m), **{f"test_{k}": v for k, v in t.items() if k not in ("per_spectrum", "per_bin")},
                      "code_size": code, "parameters": n_params(m),
                      "nonzero_parameters": int(sum(int((p != 0).sum()) for p in m.parameters())),
                      "encoder_layers": len(linears(m.encoder))}
    summary = {"rows": rows, "validation_grid": grid, "validation_evaluations": len(grid) + 2 * len(chosen),
               "chosen": {"alpha": alpha, "iters": iters, "mode": best_mode, "label": name_fb},
               "weight_report": "weight_report.json"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'model':55s} {'code':>4s} {'nonzero':>8s} {'train':>9s} {'val':>9s} {'test':>9s} {'median':>9s} "
          f"{'p95':>9s} {'p99':>9s} {'ppm med':>7s} {'ppm p99':>7s}")
    for name, r in rows.items():
        print(f"{name:55s} {r['code_size']:4d} {r['nonzero_parameters']:8d} {r['train_mse']:9.3e} {r['val_mse']:9.3e} {r['test_mse']:9.3e} "
              f"{r['test_median']:9.3e} {r['test_p95']:9.3e} {r['test_p99']:9.3e} "
              f"{r['test_rms_ppm_median']:7.2f} {r['test_rms_ppm_p99']:7.1f}")

    (out / "feedback").mkdir(exist_ok=True)
    save_programmed(out / "feedback" / "recipe.pt", {"alpha": alpha, "iters": iters, "mode": best_mode, "base": args.best,
                                                       "edits": ["output centering"]})
    plot_results(per, ["original", name_fb], splits["test"], wavelength, out / "feedback_vs_original.png")


def plot_results(per, names, test, wavelength, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    bins = np.logspace(-7, -1.5, 80)
    for name in names:
        p = per[name]["per_spectrum"]
        axes[0].hist(p, bins, histtype="step", lw=1.5, label=f"{name}\n(mean {p.mean():.2e})")
    axes[0].set(xscale="log", xlabel="per-spectrum test MSE (z-normalised)", ylabel="spectra")
    axes[0].legend(fontsize=7)
    order = np.argsort(wavelength)
    for name in names:
        axes[1].plot(wavelength[order], per[name]["per_bin"][order], "o-", ms=3, label=name)
    axes[1].set(xscale="log", yscale="log", xlabel="wavelength [µm]", ylabel="test MSE per bin")
    axes[1].legend(fontsize=7)
    a, b = per[names[0]]["per_spectrum"], per[names[1]]["per_spectrum"]
    T = test.params[:, 3]
    edges = np.quantile(T, np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(T, edges) - 1, 0, 7)
    ratio = [b[idx == i].mean() / a[idx == i].mean() for i in range(8)]
    centres = [np.median(T[idx == i]) for i in range(8)]
    axes[2].plot(centres, ratio, "o-")
    axes[2].axhline(1, color="k", ls="--", lw=1)
    axes[2].set(xlabel="T (K), octiles of the test set", ylabel="mean MSE: programmed / original",
                title="Improvement by temperature")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
