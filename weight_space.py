"""Weight-space methods on the best noiseless autoencoder (runs/ae_wide_cosine, seed 42).

The best single model is 52-256-128-64-32-16-8 (mirrored), GELU, cosine schedule, trained on
the POS spectra with split seed 42. This script works directly on its weights:

  finetune  Branches that start from the best checkpoint (all on the same train split):
              swa_*  constant learning rate; the weights are averaged every epoch (SWA) and an
                     exponential moving average (EMA) is kept as well.
              cos_*  cosine warm restart (peak lr -> 1e-7) with different batch-order seeds,
                     used as ingredients for model soups; cos_lr3e-4_long is the
                     no-averaging control with the same 1000-epoch budget as the long SWA runs.
            Branches run in parallel, one thread each.
  rebasin   Permutation alignment (weight matching, Git Re-Basin) of independently trained
            seeds onto the best model, linear-interpolation loss barriers before/after
            alignment, and the aligned weight average.
  soup      Uniform and greedy soups of the fine-tuned branches, plus every other candidate
            (SWA, EMA, branch endpoints, re-basin merge). The model with the lowest
            validation MSE is selected; test MSE is reported but never used for choosing.
  all       finetune, rebasin, soup.

Usage:
    python weight_space.py all
    python weight_space.py finetune --workers 4
    python weight_space.py rebasin --seeds runs/weight_space/seeds/seed_43/model.pt ...
    python weight_space.py soup

Everything goes to runs/weight_space/. The selected model is saved as
runs/weight_space/best/model.pt in the same format as train.py.
"""

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from data import DATA_DIR, prepare_splits
from model import ACTIVATIONS, Autoencoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BEST = "runs/ae_wide_cosine/scheduler_cosine/seed_42/model.pt"
OUT = Path("runs/weight_space")

# name, batch-order seed, learning rate, epochs, schedule
BRANCHES = [
    ("swa_lr3e-5", 101, 3e-5, 300, "const"),
    ("swa_lr1e-4", 102, 1e-4, 300, "const"),
    ("swa_lr3e-4", 103, 3e-4, 300, "const"),
    ("cos_s1", 1, 1e-4, 300, "cosine"),
    ("cos_s2", 2, 1e-4, 300, "cosine"),
    ("cos_s3", 3, 1e-4, 300, "cosine"),
    ("cos_s4", 4, 1e-4, 300, "cosine"),
    # second round: longer, higher learning rates, several batch-order seeds (soup ingredients)
    ("swa_lr3e-4_long_s1", 201, 3e-4, 1000, "const"),
    ("swa_lr3e-4_long_s2", 202, 3e-4, 1000, "const"),
    ("swa_lr3e-4_long_s3", 203, 3e-4, 1000, "const"),
    ("swa_lr1e-3_long", 204, 1e-3, 1000, "const"),
    # control: the same budget of plain fine-tuning (annealed, no averaging)
    ("cos_lr3e-4_long", 205, 3e-4, 1000, "cosine"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["finetune", "rebasin", "soup", "all"])
    p.add_argument("--best", default=BEST, help="checkpoint of the model to improve")
    p.add_argument("--seeds", nargs="+",
                   default=["runs/weight_space/seeds/seed_43/model.pt",
                            "runs/weight_space/seeds/seed_44/model.pt"],
                   help="independently trained models (same split!) to align onto --best")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--ema-decays", type=float, nargs="+", default=[0.999, 0.9999],
                   help="per-step EMA decays; ~1/(1-d) steps are averaged (120 steps = 1 epoch)")
    p.add_argument("--rerun", action="store_true", help="retrain branches that already exist")
    return p.parse_args()


# ---------------------------------------------------------------- helpers

def load_model(path):
    ck = torch.load(path, weights_only=False)
    a = ck["args"]
    model = Autoencoder(ck["n_in"], tuple(a["hidden"]), a["latent"], ACTIVATIONS[a["activation"]])
    model.load_state_dict(ck["model_state"])
    return model.eval(), ck


def load_splits(ck):
    a = ck["args"]
    split_seed = a.get("split_seed")
    splits, wavelength, _ = prepare_splits(
        a.get("data_dir", DATA_DIR), a["min_feature_height_ppm"],
        seed=a["seed"] if split_seed is None else split_seed,
        dataset=a.get("dataset", "pos"), cut_datasets=a.get("cut_datasets"))
    return splits, wavelength


def with_state(model, state):
    m = copy.deepcopy(model)
    m.load_state_dict(state)
    return m.eval()


def average(states, weights=None):
    weights = np.full(len(states), 1 / len(states)) if weights is None else np.asarray(weights)
    return {k: sum(w * s[k] for w, s in zip(weights, states)) for k in states[0]}


@torch.no_grad()
def mse(model, spectra):
    x = torch.from_numpy(spectra)
    return float(((model(x) - x) ** 2).mean())


@torch.no_grad()
def evaluate(model, splits):
    """Val/test MSE (z-normalised) plus test percentiles and RMS residual in ppm."""
    out = {"val_mse": mse(model, splits["val"].spectra)}
    t = splits["test"]
    x = torch.from_numpy(t.spectra)
    err = ((model(x) - x).numpy())
    per = (err ** 2).mean(1)
    ppm = np.sqrt(((err * t.std[:, None]) ** 2).mean(1)) * 1e6
    out.update(test_mse=float(per.mean()), test_median=float(np.median(per)),
               test_p95=float(np.percentile(per, 95)), test_p99=float(np.percentile(per, 99)),
               test_rms_ppm_median=float(np.median(ppm)), test_rms_ppm_p99=float(np.percentile(ppm, 99)))
    return out


# ---------------------------------------------------------------- fine-tuning branches

def train_branch(job):
    """Fine-tune from the best checkpoint; returns final, SWA and EMA weights plus history."""
    name, seed, lr, epochs, schedule, best, batch_size, weight_decay, ema_decays, out = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model, ck = load_model(best)
    splits, _ = load_splits(ck)
    x_train = torch.from_numpy(splits["train"].spectra)
    x_val = splits["val"].spectra

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
             if schedule == "cosine" else None)
    swa = {k: v.clone() for k, v in model.state_dict().items()}
    emas = {d: {k: v.clone() for k, v in model.state_dict().items()} for d in ema_decays}
    live = model.state_dict()  # tensors share storage with the parameters
    n_swa = 1
    swa_start = epochs // 2 if schedule == "cosine" else 0
    hist = {"epoch": [], "val": [], "val_swa": [], **{f"val_ema{d}": [] for d in ema_decays}}
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x_train), generator=gen).split(batch_size):
            b = x_train[idx]
            loss = ((model(b) - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for d, ema in emas.items():
                    for k, v in live.items():
                        ema[k].mul_(d).add_(v, alpha=1 - d)
        if sched:
            sched.step()
        model.eval()
        if epoch >= swa_start:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    swa[k].mul_(n_swa / (n_swa + 1)).add_(v, alpha=1 / (n_swa + 1))
            n_swa += 1
        if epoch % 10 == 9 or epoch == epochs - 1:
            hist["epoch"].append(epoch + 1)
            hist["val"].append(mse(model, x_val))
            hist["val_swa"].append(mse(with_state(model, swa), x_val))
            for d, ema in emas.items():
                hist[f"val_ema{d}"].append(mse(with_state(model, ema), x_val))
            print(f"{name} epoch {epoch + 1}: val {hist['val'][-1]:.3e}  swa {hist['val_swa'][-1]:.3e}  " +
                  "  ".join(f"ema{d} {hist[f'val_ema{d}'][-1]:.3e}" for d in emas), flush=True)
    hist["time_s"] = round(time.time() - t0, 1)
    states = {"final": model.state_dict(), "swa": swa, **{f"ema{d}": e for d, e in emas.items()}}
    torch.save({"states": states, "history": hist,
                "config": dict(seed=seed, lr=lr, epochs=epochs, schedule=schedule)},
               Path(out) / "branches" / f"{name}.pt")
    return name, hist


def run_finetune(args):
    out = Path(args.out)
    (out / "branches").mkdir(parents=True, exist_ok=True)
    jobs = [(*b, args.best, args.batch_size, args.weight_decay, tuple(args.ema_decays), str(out))
            for b in BRANCHES if args.rerun or not (out / "branches" / f"{b[0]}.pt").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(train_branch, jobs))
    results = {b[0]: torch.load(out / "branches" / f"{b[0]}.pt", weights_only=False)["history"]
               for b in BRANCHES}

    model, ck = load_model(args.best)
    best_val = mse(model, load_splits(ck)[0]["val"].spectra)
    groups = {"constant lr + SWA/EMA, 300 epochs": [n for n in results if n.startswith("swa") and "long" not in n],
              "1000 epochs: constant lr + SWA/EMA vs annealed control": [n for n in results if "long" in n],
              "cosine warm restart, 300 epochs (soup ingredients)": [n for n in results if n.startswith("cos_s")]}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for ax, (title, names) in zip(axes, groups.items()):
        for i, name in enumerate(names):
            h = results[name]
            ep = np.array(h["epoch"])
            if i == 0:  # raw weights are noisy; show one branch per panel
                ax.plot(ep, h["val"], color="0.6", lw=0.7, label=f"{name} weights (no averaging)")
            first = len(ep) // 2 if name.startswith("cos") else 0  # SWA starts halfway for cosine runs
            ax.plot(ep[first:], h["val_swa"][first:], color=f"C{i}", lw=2, label=f"{name} SWA")
            if name.startswith("cos"):
                ax.plot(ep, h["val"], color=f"C{i}", lw=1, ls="--", label=f"{name} final weights")
                continue
            for key, ls in (("val_ema0.999", ":"), ("val_ema0.9999", "-.")):
                if key in h:
                    ax.plot(ep, h[key], color=f"C{i}", lw=1.5, ls=ls, label=f"{name} EMA {key[7:]}")
        ax.axhline(best_val, color="k", ls="--", lw=1, label="start (best model)")
        ax.set(xlabel="fine-tuning epoch", yscale="log", title=title, ylim=(4.2e-5, 7.5e-5))
        ax.legend(fontsize=6, ncol=2, loc="upper right")
    axes[0].set_ylabel("val MSE (z-normalised)")
    fig.tight_layout()
    fig.savefig(out / "finetune_curves.png", dpi=150)
    plt.close(fig)
    (out / "finetune_history.json").write_text(json.dumps(results))


# ---------------------------------------------------------------- re-basin (weight matching)

def linear_layers(state):
    """Weight/bias keys of the Linear layers in forward order (encoder then decoder)."""
    keys = [k[:-len(".weight")] for k in state if k.endswith(".weight")]
    return sorted(keys, key=lambda k: (not k.startswith("encoder"), int(k.split(".")[1])))


def apply_perms(state, layers, perms):
    """perms[i] permutes the outputs of layer i (and inputs of layer i+1); None = identity."""
    s = {k: v.clone() for k, v in state.items()}
    for i, name in enumerate(layers):
        w, b = s[f"{name}.weight"], s[f"{name}.bias"]
        if i > 0 and perms[i - 1] is not None:
            w = w[:, perms[i - 1]]
        if perms[i] is not None:
            w, b = w[perms[i]], b[perms[i]]
        s[f"{name}.weight"], s[f"{name}.bias"] = w, b
    return s


def weight_matching(state_a, state_b, max_iter=100, seed=0):
    """Find hidden-unit permutations of model B that maximise the inner product with model A.

    Coordinate ascent over layers, each step a linear assignment (Ainsworth et al. 2023).
    Every hidden layer, including the linear 8-d bottleneck, is permutable.
    """
    layers = linear_layers(state_a)
    W = lambda s, i: s[f"{layers[i]}.weight"].double().numpy()  # noqa: E731
    B = lambda s, i: s[f"{layers[i]}.bias"].double().numpy()  # noqa: E731
    n_perm = len(layers) - 1
    perms = [np.arange(W(state_a, i).shape[0]) for i in range(n_perm)]
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        changed = False
        for i in rng.permutation(n_perm):
            wb = W(state_b, i)
            if i > 0:
                wb = wb[:, perms[i - 1]]
            cost = W(state_a, i) @ wb.T + np.outer(B(state_a, i), B(state_b, i))
            nxt = W(state_b, i + 1)
            if i + 1 < n_perm:
                nxt = nxt[perms[i + 1]]
            cost += W(state_a, i + 1).T @ nxt
            _, col = linear_sum_assignment(cost, maximize=True)
            if not np.array_equal(col, perms[i]):
                old = cost[np.arange(len(col)), perms[i]].sum()
                new = cost[np.arange(len(col)), col].sum()
                if new > old + 1e-12:
                    perms[i], changed = col, True
        if not changed:
            break
    return apply_perms(state_b, layers, [torch.from_numpy(p) for p in perms] + [None])


@torch.no_grad()
def hidden_activations(model, x):
    """Output of every hidden layer (after its activation; the bottleneck is linear)."""
    acts, h = [], torch.from_numpy(x)
    mods = list(model.encoder) + list(model.decoder)
    for i, m in enumerate(mods):
        h = m(h)
        nxt = mods[i + 1] if i + 1 < len(mods) else None
        if isinstance(m, torch.nn.Linear) and isinstance(nxt, torch.nn.Linear):
            acts.append(h.double().numpy())  # linear bottleneck
        elif not isinstance(m, torch.nn.Linear) and m is not mods[-1]:
            acts.append(h.double().numpy())
    return acts


def activation_matching(model_a, model_b, x):
    """Permute B's hidden units to maximise the correlation of their activations with A's
    on the inputs x (Ainsworth et al. 2023; Entezari et al. 2022)."""
    perms = []
    for a, b in zip(hidden_activations(model_a, x), hidden_activations(model_b, x)):
        za = (a - a.mean(0)) / (a.std(0) + 1e-8)
        zb = (b - b.mean(0)) / (b.std(0) + 1e-8)
        _, col = linear_sum_assignment(za.T @ zb, maximize=True)
        perms.append(torch.from_numpy(col))
    state_b = model_b.state_dict()
    return apply_perms(state_b, linear_layers(state_b), perms + [None])


def split_seed(a):
    return a["seed"] if a.get("split_seed") is None else a["split_seed"]


def run_rebasin(args):
    out = Path(args.out) / "rebasin"
    out.mkdir(parents=True, exist_ok=True)
    model_a, ck = load_model(args.best)
    splits, _ = load_splits(ck)
    xv = splits["val"].spectra
    x_match = splits["train"].spectra[np.random.default_rng(0).choice(len(splits["train"].spectra), 8000, False)]
    state_a = model_a.state_dict()
    lams = np.linspace(0, 1, 21)
    results = {}
    aligned = {"weight": [state_a], "activation": [state_a]}
    fig, axes = plt.subplots(1, len(args.seeds), figsize=(5.5 * len(args.seeds), 4), squeeze=False)
    for ax, path in zip(axes[0], args.seeds):
        model_b, ck_b = load_model(path)
        assert split_seed(ck_b["args"]) == split_seed(ck["args"]), \
            f"{path} was trained on a different split; merging it would leak test data"
        state_b = model_b.state_dict()
        variants = {"naive": state_b,
                    "weight": weight_matching(state_a, state_b),
                    "activation": activation_matching(model_a, model_b, x_match)}
        name = Path(path).parent.name
        res = {}
        for label, sb in variants.items():
            # A permutation must not change the function.
            assert abs(mse(with_state(model_a, sb), xv) - mse(model_b, xv)) < 1e-9
            if label != "naive":
                aligned[label].append(sb)
            curve = [mse(with_state(model_a, average([state_a, sb], [1 - lam, lam])), xv) for lam in lams]
            ends = (1 - lams) * curve[0] + lams * curve[-1]
            res[label] = {"val_curve": curve, "barrier": float(np.max(np.array(curve) - ends)),
                          "midpoint_val": curve[len(lams) // 2]}
            ax.plot(lams, curve, label=f"{label} (barrier {res[label]['barrier']:.2g})")
            print(f"{name} {label:10s}: barrier {res[label]['barrier']:.3e}, midpoint val {curve[10]:.3e}")
        res["endpoint_val"] = [res["naive"]["val_curve"][0], res["naive"]["val_curve"][-1]]
        results[name] = res
        ax.set(xlabel=f"λ (0 = best model, 1 = {name})", ylabel="val MSE (z-normalised)", yscale="log",
               title=f"seed 42 ↔ {name}: linear interpolation")
        ax.legend(fontsize=8, title="matching")
    fig.tight_layout()
    fig.savefig(out / "interpolation.png", dpi=150)
    plt.close(fig)

    merged = {k: average(v) for k, v in aligned.items()}
    for k, m in merged.items():
        results[f"merged_{k}_matching_val"] = mse(with_state(model_a, m), xv)
        print(f"{len(aligned[k])}-model average after {k} matching: val {results[f'merged_{k}_matching_val']:.3e}")
    best_kind = min(merged, key=lambda k: results[f"merged_{k}_matching_val"])
    torch.save({"aligned": aligned, "merged": merged[best_kind], "matching": best_kind}, out / "aligned.pt")
    (out / "rebasin_results.json").write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------- soups and selection

def greedy_soup(model, states, xv):
    """Add ingredients (sorted by val) to the soup only if the val MSE of the average improves."""
    order = np.argsort([mse(with_state(model, s), xv) for s in states])
    chosen = [order[0]]
    best = mse(with_state(model, states[order[0]]), xv)
    for i in order[1:]:
        trial = mse(with_state(model, average([states[j] for j in chosen + [i]])), xv)
        if trial < best:
            chosen.append(i)
            best = trial
    return average([states[j] for j in chosen]), [int(j) for j in chosen]


def run_soup(args):
    out = Path(args.out)
    model, ck = load_model(args.best)
    splits, wavelength = load_splits(ck)
    xv = splits["val"].spectra
    branches = {p.stem: torch.load(p, weights_only=False) for p in sorted((out / "branches").glob("*.pt"))}

    cands = {"original (best seed 42)": model.state_dict()}
    for name, b in branches.items():
        cands[f"{name} final"] = b["states"]["final"]
        cands[f"{name} SWA" + (" (2nd half)" if name.startswith("cos") else "")] = b["states"]["swa"]
        if name.startswith("cos"):
            continue  # EMA of an annealed run is the same as its final weights
        for key in b["states"]:
            if key.startswith("ema"):
                cands[f"{name} EMA {key[3:]}"] = b["states"][key]
    cos_names = [n for n in branches if n.startswith("cos_s")]
    long_names = [n for n in branches if n.startswith("swa") and "long" in n]
    same_lr = [n for n in long_names if n.startswith("swa_lr3e-4_long")]
    if same_lr:
        # these share a basin; branches with a different lr (1e-3) end up in another one
        cands["uniform soup (lr 3e-4 long-run EMAs 0.999)"] = average(
            [branches[n]["states"]["ema0.999"] for n in same_lr])
    ingredients = [branches[n]["states"]["final"] for n in cos_names]
    if ingredients:
        cands["uniform soup (cos finals)"] = average(ingredients)
        cands["uniform soup (cos SWAs)"] = average([branches[n]["states"]["swa"] for n in cos_names])
        pool = (ingredients + [branches[n]["states"]["swa"] for n in cos_names] + [model.state_dict()]
                + [branches[n]["states"][k] for n in long_names for k in ("ema0.999", "ema0.9999")])
        pool_names = ([f"{n} final" for n in cos_names] + [f"{n} SWA" for n in cos_names] + ["original"]
                      + [f"{n} {k}" for n in long_names for k in ("ema0.999", "ema0.9999")])
        soup, chosen = greedy_soup(model, pool, xv)
        cands["greedy soup"] = soup
        greedy_members = [pool_names[j] for j in chosen]
    rebasin = out / "rebasin" / "aligned.pt"
    if rebasin.exists():
        rb = torch.load(rebasin, weights_only=False)
        cands[f"re-basin merge ({len(rb['aligned'][rb['matching']])} seeds, {rb['matching']} matching)"] = rb["merged"]

    rows = {name: evaluate(with_state(model, s), splits) for name, s in cands.items()}
    selected = min(rows, key=lambda n: rows[n]["val_mse"])
    summary = {"rows": rows, "selected": selected, "selected_by": "val_mse"}
    if ingredients:
        summary["greedy_soup_members"] = greedy_members
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{'candidate':40s} {'val':>9s} {'test':>9s} {'median':>9s} {'p95':>9s} {'p99':>9s} {'ppm med':>7s} {'ppm p99':>7s}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["val_mse"]):
        print(f"{name:40s} {r['val_mse']:9.3e} {r['test_mse']:9.3e} {r['test_median']:9.3e} "
              f"{r['test_p95']:9.3e} {r['test_p99']:9.3e} {r['test_rms_ppm_median']:7.2f} {r['test_rms_ppm_p99']:7.1f}")
    print(f"selected (lowest val MSE): {selected}")

    best_dir = out / "best"
    best_dir.mkdir(exist_ok=True)
    final = with_state(model, cands[selected])
    torch.save({"model_state": final.state_dict(), "args": {**ck["args"], "weight_space": selected},
                "n_in": ck["n_in"]}, best_dir / "model.pt")
    (best_dir / "metrics.json").write_text(json.dumps({"selected": selected, **rows[selected]}, indent=2))
    plot_summary(rows, selected, out / "candidates.png")
    plot_error_hist(model, final, splits, selected, out / "test_error_hist.png")


def plot_summary(rows, selected, path):
    names = sorted(rows, key=lambda n: rows[n]["val_mse"])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.2))
    ax.barh(y - 0.2, [rows[n]["val_mse"] for n in names], 0.4, label="val")
    ax.barh(y + 0.2, [rows[n]["test_mse"] for n in names], 0.4, label="test")
    ax.axvline(rows["original (best seed 42)"]["test_mse"], color="k", ls="--", lw=1, label="original test")
    ax.set_yticks(y, [f"{n}{'  ◀' if n == selected else ''}" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="MSE (z-normalised)", xscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_error_hist(original, final, splits, label, path):
    x = torch.from_numpy(splits["test"].spectra)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(-7, -1.5, 80)
    for m, name in ((original, "original"), (final, label)):
        per = ((m(x) - x) ** 2).mean(1).numpy()
        ax.hist(per, bins, histtype="step", lw=1.5, label=f"{name} (mean {per.mean():.2e})")
    ax.set(xscale="log", xlabel="per-spectrum test MSE (z-normalised)", ylabel="spectra")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.stage in ("finetune", "all"):
        run_finetune(args)
    if args.stage in ("rebasin", "all"):
        run_rebasin(args)
    if args.stage in ("soup", "all"):
        run_soup(args)


if __name__ == "__main__":
    main()
