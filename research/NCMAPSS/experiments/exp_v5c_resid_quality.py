"""
exp_v5c_resid_quality.py — residual-quality diagnostic: proposed
healthy-range GP (v5c) vs official cycle<=3 GP at the SAME subsampled
points (both stats follow the u*7+sd rng convention -> exact pairing).

Dev part: repeats the 2026-08-15 session diagnostic (cached dev stats).
Test part = the FOURTEENTH DS03 test opening, pre-registered BEFORE
looking: metric definitions identical to dev, cached stats only, no
model contact, all numbers reported regardless of direction.

Metrics per unit-seed case (trim25 per-cycle tables; channels scaled by
the official GP's pooled healthy sigma of the same split/seed):
  floor  = RMS over healthy cycles (cyc < TRUE onset; ground truth used
           for diagnostics only — never for training or selection)
  signal = mean |row| over the last 5 cycles
  snr    = signal / floor
Paired Wilcoxon official vs v5c on floor and snr; win counts.
Outputs: v5c_resid_quality_results.txt, v5c_resid_quality.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_hi import trim25

SEEDS = [0, 1, 2]
NEND = 5
RES = os.path.join(HERE, 'v5c_resid_quality_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def tab(Z, u):
    cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']
    ucyc = np.unique(cc)
    return ucyc, np.stack([trim25(rs[cc == c]) for c in ucyc])


def split_metrics(name, off_fmt, v5c_fmt):
    fl_o, fl_c, sg_o, sg_c, ids = [], [], [], [], []
    for sd in SEEDS:
        Zo = np.load(os.path.join(HERE, off_fmt.format(sd=sd)))
        Zc = np.load(os.path.join(HERE, v5c_fmt.format(sd=sd)))
        units = sorted({int(k[1:].split('_')[0]) for k in Zo.files
                        if k.endswith('_cc')})
        pool = []
        for u in units:
            assert np.array_equal(Zo[f'u{u}_cc'], Zc[f'u{u}_cc']), \
                f'{name} seed{sd} u{u}: subsample mismatch — pairing broken'
            on = int(Zo[f'u{u}_onset'][0])
            uc, To = tab(Zo, u)
            pool.append(To[uc < on])
        sig_ref = np.concatenate(pool).std(0) + 1e-12
        for u in units:
            on = int(Zo[f'u{u}_onset'][0])
            uc, To = tab(Zo, u); _, Tc = tab(Zc, u)
            h = uc < on
            To = To / sig_ref; Tc = Tc / sig_ref
            fl_o.append(float(np.sqrt((To[h] ** 2).mean())))
            fl_c.append(float(np.sqrt((Tc[h] ** 2).mean())))
            sg_o.append(float(np.abs(To[-NEND:]).mean()))
            sg_c.append(float(np.abs(Tc[-NEND:]).mean()))
            ids.append((sd, u))
    fl_o, fl_c = np.array(fl_o), np.array(fl_c)
    sg_o, sg_c = np.array(sg_o), np.array(sg_c)
    n = len(fl_o)
    log(f'\n[{name}] {n} unit-seed cases '
        '(yardstick: official healthy sigma per channel, per seed)')
    log(f'  healthy floor RMS : official {fl_o.mean():.3f}  v5c '
        f'{fl_c.mean():.3f}  ratio {np.mean(fl_c / fl_o):.2f}  '
        f'p={st.wilcoxon(fl_o, fl_c).pvalue:.1e}  '
        f'v5c lower in {int((fl_c < fl_o).sum())}/{n}')
    log(f'  end-of-life signal: official {sg_o.mean():.3f}  v5c '
        f'{sg_c.mean():.3f}  ratio {np.mean(sg_c / sg_o):.2f}')
    r_o, r_c = sg_o / fl_o, sg_c / fl_c
    log(f'  SNR signal/floor  : official {r_o.mean():.2f}  v5c '
        f'{r_c.mean():.2f}  p={st.wilcoxon(r_o, r_c).pvalue:.1e}  '
        f'v5c better in {int((r_c > r_o).sum())}/{n}')
    per_u = {}
    for (sd, u), a, b in zip(ids, sg_o / fl_o, sg_c / fl_c):
        per_u.setdefault(u, []).append(b / a)
    log('  per-unit SNR gain (v5c/official, mean over seeds): '
        + '  '.join(f'u{u} {np.mean(v):.2f}x' for u, v in sorted(per_u.items())))
    return dict(fl_o=fl_o, fl_c=fl_c, sg_o=sg_o, sg_c=sg_c,
                ids=np.array(ids, int))


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C RESIDUAL QUALITY — official cycle<=3 GP vs proposed healthy-'
        'range GP, paired points.  Test part = FOURTEENTH DS03 opening, '
        'pre-registered (metrics frozen from the dev diagnostic).')
    out = {}
    out['dev'] = split_metrics('dev', 'v4_c3_stats_s{sd}.npz',
                               'v5c_stats_s{sd}.npz')
    out['test'] = split_metrics('test', 'v4_test_stats_s{sd}.npz',
                                'v5c_test_stats_s{sd}.npz')
    np.savez(os.path.join(HERE, 'v5c_resid_quality.npz'),
             **{f'{s}_{k}': v for s in out for k, v in out[s].items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
