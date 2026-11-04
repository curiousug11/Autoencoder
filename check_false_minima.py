"""Step 2d: are the poor-fit secondary minima from random starts genuine local minima?

In the noiseless first pass, 6-35% of random-start runs end at a second "minimum" with a much
worse fit (separate, not credible). Here each of those endpoints is descended further (to 300,
900 and 3000 more Levenberg-Marquardt steps) to tell genuine poor-fit local minima from
descents that were still crawling when the step budget ran out.

    python check_false_minima.py      # writes runs/mode_paths/false_minima_check.json
"""
import json, sys
from types import SimpleNamespace
import numpy as np, torch

from mode_paths import CODES, Chi2, classify, credible, descend, iter_sigmas, prepare
torch.set_num_threads(4)
run = 'runs/mode_paths'
args = SimpleNamespace(**json.load(open(f'{run}/summary.json'))['args'])
out = {}
checkpoints = [300, 900, 3000]
for code in args.codes:
    prep = prepare(code, args)
    rng = np.random.default_rng([args.seed, CODES.index(code)])
    for sigma, _, _, _, sets in iter_sigmas(prep, args, rng):
        z = np.load(f'{run}/{code}_sigma{sigma:g}.npz')
        r = {k.split('_', 1)[1]: z[k] for k in z.files if k.startswith('random_')}
        _, sep, _, _ = classify(r, args)
        q = np.flatnonzero(sep & ~credible(r))
        if not len(q):
            continue
        f = Chi2(prep['models'], prep['lo'], prep['hi'], sets['random'][3][q], sigma * 1e-6)
        ub = torch.as_tensor(r['ub'][q]); done = 0; hist = {}
        for cp in checkpoints:
            ub = descend(f, ub, cp - done); done = cp
            hist[cp] = f(ub[:, None])[:, 0].numpy()
        c_end = hist[checkpoints[-1]]
        # still descending: chi2 fell by >1% over the last 2100 steps
        moving = c_end < 0.99 * hist[900]
        poor = c_end > 16.9
        e = dict(n=int(len(q)), good_fit=int((~poor).sum()), poor_still_descending=int((poor & moving).sum()),
                 poor_plateau=int((poor & ~moving).sum()),
                 plateau_chi2=np.round(c_end[poor & ~moving], 1).tolist())
        out[f'{code}_{sigma:g}'] = e
        print(code, sigma, e, flush=True)
tot = {k: sum(v[k] for v in out.values()) for k in ('n', 'good_fit', 'poor_still_descending', 'poor_plateau')}
print('total', tot)
json.dump({'per_code_sigma': out, 'total': tot, 'descent_steps_checkpoints': checkpoints,
           'note': 'random-start secondary minima classed separate-not-credible in the noiseless first pass, '
                   'descended further from the saved endpoint'}, open(f'{run}/false_minima_check.json', 'w'), indent=1)
