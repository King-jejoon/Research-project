"""
exp_rul_r5.py — R5: ONE-TIME pure verification on the sealed DS03 test units.

Frozen chain (all calibrated on dev only):
  detection: gated q75-LL (V* absolute, from R1) -> 30 flight-hour baseline
             -> clip mu0-10*sd0 -> full-range mean-drop     [dev RMSE 4.72]
  RUL      : trim residual summary -> dev z-norm -> HI MLP (HI_CFG,
             end_target=1.03) trained on all 9 dev units -> Stage-5 cycle-axis
             exponential first-passage, beta by dev LOO CV  [dev LOO 7.36]
Outputs: test onset RMSE (6 units x 3 seeds) and test truncation RUL RMSE/NASA
         (6 units x 4 fracs x 3 seeds) vs official 7.23 +/- 0.10 / 17.8.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from neural_fusion_tail import train_model_tail
from exp_rul_r23 import (nasa, eval_units, HI_CFG, FRACS, BETA_C, GRID)

SEEDS = [0, 1, 2]
VARIANT = 'trim'
RES = os.path.join(HERE, 'rul_r5_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load(sd):
    z = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    out = {}
    for split in ['dev', 'test']:
        units = sorted({int(k.split('_')[1]) for k in z.files
                        if k.startswith(f'{split}_') and k.endswith('_ucyc')})
        out[split] = {u: dict(raw=z[f'{split}_{u}_{VARIANT}'],
                              hours=np.cumsum(z[f'{split}_{u}_hours']),
                              dur=z[f'{split}_{u}_hours'],
                              llq=z[f'{split}_{u}_llq75'],
                              onset=int(z[f'{split}_{u}_onset'][0]),
                              ucyc=z[f'{split}_{u}_ucyc']) for u in units}
    return out


def meandrop(x, kmin=3):
    n = len(x); s = np.std(x, ddof=1) + 1e-12; cs = np.cumsum(x)
    best = (None, -np.inf)
    for k in range(kmin, n - 4):
        m1 = cs[k - 1] / k; m2 = (cs[-1] - cs[k - 1]) / (n - k)
        t = np.sqrt(k * (n - k) / n) * (m1 - m2) / s
        if t > best[1]:
            best = (k, t)
    return best[0]


def detect(d):
    cur = d['llq']; dur = d['dur']
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(d['ucyc'][k]) if k < len(d['ucyc']) else int(d['ucyc'][-1] + 1)


def main():
    open(RES, 'w').close()
    log('R5 PURE TEST VERIFICATION  (frozen chain, one-time)')
    t0 = time.time()
    cfg = dict(HI_CFG); cfg['end_target'] = 1.03

    det_d = []; allP, allT, allF = [], [], []
    for sd in SEEDS:
        D = load(sd)
        dev, tst = D['dev'], D['test']
        # ---- detection on test ----
        row = []
        for u in sorted(tst):
            det = detect(tst[u])
            row.append((u, tst[u]['onset'], det, det - tst[u]['onset']))
            det_d.append(det - tst[u]['onset'])
        log(f'  seed{sd} onset: ' + '  '.join(f'u{u}:t{t}/d{dd}({e:+d})'
                                              for u, t, dd, e in row))
        # ---- RUL ----
        du = sorted(dev)
        allr = np.concatenate([dev[u]['raw'] for u in du])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zd = {u: (dev[u]['raw'] - mu) / sg for u in du}
        Zt = {u: (tst[u]['raw'] - mu) / sg for u in sorted(tst)}
        np.random.seed(sd)
        him, _ = train_model_tail([Zd[u] for u in du], **cfg)
        dHI = {u: him.forward(Zd[u]).flatten() for u in du}
        tHI = {u: him.forward(Zt[u]).flatten() for u in sorted(tst)}
        cd = {u: np.arange(len(dHI[u])) / 500.0 for u in du}
        ct = {u: np.arange(len(tHI[u])) / 500.0 for u in tHI}
        # beta by dev LOO (NASA), frozen protocol
        best = None
        for b in BETA_C:
            tot = 0.0
            for u in du:
                tr = [v for v in du if v != u]
                P, T, _, _ = eval_units(b, {v: cd[v] for v in tr},
                                        {v: dHI[v] for v in tr},
                                        {u: cd[u]}, {u: dHI[u]}, FRACS, None)
                tot += nasa(P, T)
            if best is None or tot < best[1]:
                best = (b, tot)
        beta = best[0]
        P, T, F, U = eval_units(beta, cd, dHI, ct, tHI, FRACS, None)
        allP.append(P); allT.append(T); allF.append(F)
        rmse = float(np.sqrt(((P - T) ** 2).mean()))
        log(f'  seed{sd} RUL: beta={beta}  test RMSE={rmse:.2f}  '
            f'NASA={nasa(P, T):.1f}  ({time.time()-t0:.0f}s)')

    # ---- summary ----
    d = np.array(det_d, float)
    log('')
    log(f'ONSET (test 6u x 3sd): RMSE={np.sqrt((d**2).mean()):.2f} cyc  '
        f'mean delay={d.mean():+.2f}  |d|<=3: {100*(np.abs(d)<=3).mean():.0f}%  '
        f'(dev was 4.72)')
    rs = [float(np.sqrt(((P - T) ** 2).mean())) for P, T in zip(allP, allT)]
    scs = [nasa(P, T) for P, T in zip(allP, allT)]
    log(f'RUL   (test 6u x 4fr x 3sd): RMSE={np.mean(rs):.2f} +/- {np.std(rs):.2f}  '
        f'NASA={np.mean(scs):.1f} +/- {np.std(scs):.1f}  '
        f'(official baseline 7.23 +/- 0.10 / 17.8)')
    log(f'{"frac":>6} | {"RMSE":>6} | {"NASA":>6}')
    for f in FRACS:
        r = [float(np.sqrt(((P[F == f] - T[F == f]) ** 2).mean()))
             for P, T, F in zip(allP, allT, allF)]
        s = [nasa(P[F == f], T[F == f]) for P, T, F in zip(allP, allT, allF)]
        log(f'{f:>6.0%} | {np.mean(r):>6.2f} | {np.mean(s):>6.1f}')
    log(f'wall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
