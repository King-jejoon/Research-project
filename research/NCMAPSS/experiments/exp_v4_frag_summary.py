"""
exp_v4_frag_summary.py — downstream-fragility grid: what does residual
quality buy in RUL terms once the downstream armour is removed?

Motivation: the truncation-RUL tie across normal models is conditional on a
heavily robust downstream (trim25 summary + med3 filter) that structurally
removes competitor artefacts (LLKE x100-300 spikes pass trim25 at 0.03-0.13%).
This grid swaps the armour off one piece at a time, everything else frozen:

  summary  in {trim25, mean}      x   HI post-filter in {med3, none}
  models: mogp gated V*=20.2 / mogp ungated / llke / bspline / lr / cabn
  score : dev LOO truncation RUL RMSE + NASA + shape gate, 3 seeds

trim25+med3 is the reference cell and must reproduce the known numbers
(mogp 7.30, llke 7.63, bspline 7.43, lr=cabn 7.77) — sanity gate.
All dev; no test opening.  Sources: v4_c3_stats_s*.npz (mogp points),
v4_bench_resid_s*.npz (competitor residuals), rul_input_s*.npz (hours).
Outputs: v4_frag_summary_results.txt, v4_frag_summary.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_hi import trim25, SEEDS, VSTAR
from exp_v4_final import med3

L1, L2 = 8.0, 0.25
VOFF = -1e9
RES = os.path.join(HERE, 'v4_frag_summary_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def summarize(rows, how):
    return trim25(rows) if how == 'trim25' else rows.mean(0)


def mogp_tables(sd, V, how):
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            T[i] = summarize(rs[m if m.any() else b], how)
        raw[u] = T
        hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    return units, raw, hrs


def bench_tables(sd, name, how):
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
    return sorted(raw), raw, hrs


def run(loader, how, filt):
    rms, nss, oks = [], [], []
    for sd in SEEDS:
        units, raw, hrs = loader(sd, how)
        allr = np.concatenate([raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn = {u: (raw[u] - mu) / sg for u in units}
        np.random.seed(sd)
        cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2,
                                       end_target=1.03)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        post = med3 if filt == 'med3' else (lambda h: h)
        HIs = {u: post(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok))
    return (float(np.mean(rms)), float(np.std(rms)),
            float(np.mean(nss)), int(np.sum(oks)))


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('DOWNSTREAM FRAGILITY GRID — summary {trim25, mean} x filter '
        '{med3, none}, frozen HI recipe, dev LOO, 3 seeds')
    loaders = {
        'mogp gated': lambda sd, how: mogp_tables(sd, VSTAR, how),
        'mogp ungated': lambda sd, how: mogp_tables(sd, VOFF, how),
        'llke': lambda sd, how, n='llke': bench_tables(sd, n, how),
        'bspline': lambda sd, how, n='bspline': bench_tables(sd, n, how),
        'lr': lambda sd, how, n='lr': bench_tables(sd, n, how),
        'cabn': lambda sd, how, n='cabn': bench_tables(sd, n, how),
    }
    out = {}
    for how in ['trim25', 'mean']:
        for filt in ['med3', 'none']:
            log(f'\n--- summary={how}  filter={filt} ---')
            for name, ld in loaders.items():
                r = run(ld, how, filt)
                out[f'{name}|{how}|{filt}'] = r
                log(f'  {name:>13}: RUL {r[0]:6.2f} ± {r[1]:4.2f}  '
                    f'NASA {r[2]:8.1f}  shape {r[3]}/3')
    log('\nGRID SUMMARY — dev LOO RUL RMSE (shape)')
    hdr = f'  {"model":>13} | trim25+med3 | trim25 only | mean+med3 | mean only'
    log(hdr)
    for name in loaders:
        cells = []
        for how in ['trim25', 'mean']:
            for filt in ['med3', 'none']:
                r = out[f'{name}|{how}|{filt}']
                cells.append(f'{r[0]:6.2f} {r[3]}/3')
        log(f'  {name:>13} | ' + ' | '.join(cells))
    np.savez(os.path.join(HERE, 'v4_frag_summary.npz'),
             **{k.replace(' ', '_').replace('|', '__'): np.array(v)
                for k, v in out.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
