"""exp_bench2_rul.py — the benchmark table: RUL RMSE / NASA Score / running time.

Each competing normal model (lr / bspline / llke / cabn) is put in the MOGP slot;
its residual table (bench2_{m}_s{sd}.npz, built by exp_bench2_r1.py) is pushed
through the FROZEN downstream, identical to exp_rul_r5.py:

  onset : gated q75-LL -> 30 flight-hour baseline -> clip mu0-10*sd0
          -> full-range mean-drop change point
  RUL   : trim residual -> dev z-norm -> HI MLP (HI_CFG, end_target 1.03)
          -> cycle-axis exponential first-passage, beta by dev LOO NASA

`mogp` reads the untouched rul_input_s{sd}.npz, so our own row is the frozen
pipeline itself.  Metrics: dev LOO RUL, test RUL (RMSE + NASA), onset RMSE,
and the running time of the normal-model stage.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from neural_fusion_tail import train_model_tail
from exp_rul_r23 import nasa, eval_units, HI_CFG, FRACS, BETA_C
from exp_rul_r5 import detect

SEEDS = [0, 1, 2]
VARIANT = 'trim'
METHODS = ['mogp', 'llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'bench2_rul_results.txt')
GP_COST = dict(fit=775.0, infer_dev=237.3, infer_test=136.0)   # from rul_r1 log


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load(method, sd):
    fn = f'rul_input_s{sd}.npz' if method == 'mogp' else f'bench2_{method}_s{sd}.npz'
    z = np.load(os.path.join(HERE, fn))
    out = {}
    for split in ['dev', 'test']:
        units = sorted({int(k.split('_')[1]) for k in z.files
                        if k.startswith(f'{split}_') and k.endswith('_ucyc')})
        out[split] = {u: dict(raw=z[f'{split}_{u}_{VARIANT}'],
                              dur=z[f'{split}_{u}_hours'],
                              llq=z[f'{split}_{u}_llq75'],
                              onset=int(z[f'{split}_{u}_onset'][0]),
                              ucyc=z[f'{split}_{u}_ucyc']) for u in units}
    return out


def run_method(method):
    cfg = dict(HI_CFG); cfg['end_target'] = 1.03
    det_dev, det_tst = [], []
    devP, devT, devF, tstP, tstT, tstF = [], [], [], [], [], []
    betas = []
    for sd in SEEDS:
        D = load(method, sd)
        dev, tst = D['dev'], D['test']
        for u in sorted(dev):
            det_dev.append(detect(dev[u]) - dev[u]['onset'])
        for u in sorted(tst):
            det_tst.append(detect(tst[u]) - tst[u]['onset'])

        du = sorted(dev)
        allr = np.concatenate([dev[u]['raw'] for u in du])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zd = {u: (dev[u]['raw'] - mu) / sg for u in du}
        Zt = {u: (tst[u]['raw'] - mu) / sg for u in sorted(tst)}
        np.random.seed(sd)
        him, _ = train_model_tail([Zd[u] for u in du], **cfg)
        dHI = {u: him.forward(Zd[u]).flatten() for u in du}
        tHI = {u: him.forward(Zt[u]).flatten() for u in Zt}
        cd = {u: np.arange(len(dHI[u])) / 500.0 for u in du}
        ct = {u: np.arange(len(tHI[u])) / 500.0 for u in tHI}

        best = None                                     # beta by dev LOO NASA
        for b in BETA_C:
            tot, PP, TT, FF = 0.0, [], [], []
            for u in du:
                tr = [v for v in du if v != u]
                P, T, F, _ = eval_units(b, {v: cd[v] for v in tr},
                                        {v: dHI[v] for v in tr},
                                        {u: cd[u]}, {u: dHI[u]}, FRACS, None)
                tot += nasa(P, T); PP.append(P); TT.append(T); FF.append(F)
            if best is None or tot < best[1]:
                best = (b, tot, np.concatenate(PP), np.concatenate(TT),
                        np.concatenate(FF))
        beta = best[0]; betas.append(beta)
        devP.append(best[2]); devT.append(best[3]); devF.append(best[4])
        P, T, F, _ = eval_units(beta, cd, dHI, ct, tHI, FRACS, None)
        tstP.append(P); tstT.append(T); tstF.append(F)

    def summarise(Ps, Ts, Fs):
        rm = [float(np.sqrt(((p - t) ** 2).mean())) for p, t in zip(Ps, Ts)]
        sc = [nasa(p, t) for p, t in zip(Ps, Ts)]
        pf = {f: float(np.mean([np.sqrt(((p[fr == f] - t[fr == f]) ** 2).mean())
                                for p, t, fr in zip(Ps, Ts, Fs)])) for f in FRACS}
        return float(np.mean(rm)), float(np.std(rm)), float(np.mean(sc)), \
            float(np.std(sc)), pf

    dv = np.array(det_dev, float); dt = np.array(det_tst, float)
    return dict(
        beta=betas,
        onset_dev=float(np.sqrt((dv ** 2).mean())),
        onset_test=float(np.sqrt((dt ** 2).mean())),
        onset_bias=float(dt.mean()),
        dev=summarise(devP, devT, devF),
        test=summarise(tstP, tstT, tstF),
        maxpred=float(max(p.max() for p in tstP)))


def main():
    open(RES, 'w').close()
    log('BENCHMARK: competing normal models in the MOGP slot, frozen downstream')
    log(f'  seeds={SEEDS}  fracs={FRACS}  detector and HI network frozen')
    log('  `mogp` = the frozen pipeline itself (rul_input_s*.npz, untouched)\n')
    t0 = time.time()
    times = json.load(open(os.path.join(HERE, 'bench2_times.json')))
    out = {}
    for m in METHODS:
        out[m] = run_method(m)
        log(f'  {m:>8} done  beta={out[m]["beta"]}  ({time.time()-t0:.0f}s)')

    log('')
    log(f'{"model":>8} | {"onset dev":>9} {"onset test":>10} | '
        f'{"dev LOO RUL":>13} | {"test RUL":>13} | {"test NASA":>13}')
    log('-' * 92)
    for m in METHODS:
        r = out[m]
        dr, ds, dn, _, _ = r['dev']
        tr_, ts, tn, tns, _ = r['test']
        log(f'{m:>8} | {r["onset_dev"]:9.2f} {r["onset_test"]:10.2f} | '
            f'{dr:6.2f}+/-{ds:4.2f} | {tr_:6.2f}+/-{ts:4.2f} | {tn:6.1f}+/-{tns:4.1f}')

    log('')
    log(f'{"model":>8} | ' + ' '.join(f'{f:>7.0%}' for f in FRACS) + '   (test RMSE)')
    log('-' * 52)
    for m in METHODS:
        log(f'{m:>8} | ' + ' '.join(f'{out[m]["test"][4][f]:7.2f}' for f in FRACS))

    log('')
    log('RUNNING TIME of the normal-model stage (seconds per seed, measured)')
    log(f'{"model":>8} | {"training":>9} {"dev infer":>10} {"test infer":>11} | '
        f'{"total":>9}')
    log('-' * 58)
    for m in METHODS:
        c = GP_COST if m == 'mogp' else dict(
            fit=float(np.mean(times[m]['fit'])),
            infer_dev=float(np.mean(times[m]['infer_dev'])),
            infer_test=float(np.mean(times[m]['infer_test'])))
        log(f'{m:>8} | {c["fit"]:9.1f} {c["infer_dev"]:10.1f} {c["infer_test"]:11.1f} | '
            f'{sum(c.values()):9.1f}')
    log('  (mogp timings come from the rul_r1 build log; NTHREADS=10, idle machine)')
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
