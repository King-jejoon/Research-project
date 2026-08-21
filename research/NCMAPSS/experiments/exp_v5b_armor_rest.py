"""
exp_v5b_armor_rest.py — armor-removal grid, remaining rows on the re-frozen
recipe (lambda1 = 12, lambda2 = 0.25): MOGP ungated + the four competitors,
cells {trim25, mean} x {med3, none}.  (The MOGP conditional row is in
exp_v5b_armor_cold_bench.)  Dev only, cached stats.
Outputs: v5b_armor_rest_results.txt, v5b_armor_rest.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, loo, train_model_tail, shape_ok
from exp_v4_hi import trim25, SEEDS
from exp_v4_final import med3

L1, L2 = 12.0, 0.25
BENCH = ['llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'v5b_armor_rest_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def summarize(rows, how):
    return trim25(rows) if how == 'trim25' else rows.mean(0)


def mogp_ungated(sd, how):
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = summarize(rs[cc == c], how)
        raw[u] = T
        hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u] - mu) / sg for u in units}, hrs


def bench(sd, name, how):
    Zd = np.load(os.path.join(HERE, f'v4_bench_resid_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    hours = {int(k.split('_')[1]): H[k] for k in H.files
             if k.startswith('dev_') and k.endswith('_hours')}
    cc, uu, rs = Zd['cc'], Zd['uu'], Zd[f'{name}_resid']
    raw, hrs = {}, {}
    for u in np.unique(uu):
        m = uu == u
        ucyc = np.unique(cc[m])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = summarize(rs[m & (cc == c)], how)
        raw[int(u)] = T
        hrs[int(u)] = np.cumsum(hours[int(u)][:len(ucyc)])
    units = sorted(raw)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u] - mu) / sg for u in units}, hrs


def cellv(loader, how, filt):
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    rms, oks = [], []
    for sd in SEEDS:
        units, Zn, hrs = loader(sd, how)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: him.forward(Zn[u]).flatten() for u in units}
        if filt == 'med3':
            HIs = {u: med3(h) for u, h in HIs.items()}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        oks.append(int(ok))
    return float(np.mean(rms)), float(np.std(rms)), int(np.sum(oks))


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B ARMOR REST — ungated + competitors, l12 downstream')
    out = {}
    rows = [('mogp_ungated', mogp_ungated)] + \
           [(n, (lambda sd, how, n=n: bench(sd, n, how))) for n in BENCH]
    for tag, ld in rows:
        vals = []
        for how in ['trim25', 'mean']:
            for filt in ['med3', 'none']:
                r = cellv(ld, how, filt)
                out[f'{tag}_{how}_{filt}'] = r
                vals.append(f'{how}+{filt}: {r[0]:.2f}±{r[1]:.2f} ({r[2]}/3)')
        log(f'  {tag:>12}: ' + ' | '.join(vals))
    np.savez(os.path.join(HERE, 'v5b_armor_rest.npz'),
             **{k: np.array(v) for k, v in out.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
