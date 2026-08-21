"""
exp_v5c_bench_quality.py — cross-model residual-quality numbers,
3-seed formalization, DEV ONLY (the seed-0 session diagnostic promoted
to a preserved script; the test extension is 15th-opening scope and is
NOT run here).

Metrics per unit-seed case (27 cases), identical to exp_v5c_resid_quality:
trim25 per-cycle tables, channels scaled by the stage-1 (mogp) healthy
sigma of the same seed; floor = RMS over healthy cycles (cyc < TRUE
onset, diagnostics only); signal = mean |row| over the last 5 cycles;
SNR = signal / floor.
Models: proposed (v5c stats), mogp (v4_c3 stats), llke / bspline / lr
(v4_bench_resid_s*, per-seed; cabn is residual-identical to lr).
Outputs: v5c_bench_quality_results.txt, v5c_bench_quality.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_hi import trim25

SEEDS = [0, 1, 2]
NEND = 5
BENCH = ['llke', 'bspline', 'lr']
RES = os.path.join(HERE, 'v5c_bench_quality_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def tabs_stats(path, units):
    Z = np.load(os.path.join(HERE, path))
    out = {}
    for u in units:
        cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']
        ucyc = np.unique(cc)
        out[u] = (ucyc, np.stack([trim25(rs[cc == c]) for c in ucyc]))
    return out


def tabs_pool(path, key, units):
    Z = np.load(os.path.join(HERE, path))
    cc, uu, rs = Z['cc'], Z['uu'], Z[key]
    out = {}
    for u in units:
        m = uu == u
        ucyc = np.unique(cc[m])
        out[u] = (ucyc, np.stack([trim25(rs[m & (cc == c)]) for c in ucyc]))
    return out


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C BENCH QUALITY — cross-model floor/SNR, dev 27 unit-seed cases, '
        '3 seeds (yardstick: stage-1 mogp healthy sigma per seed)')
    acc = {}
    for sd in SEEDS:
        Zo = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        units = sorted({int(k[1:].split('_')[0]) for k in Zo.files
                        if k.endswith('_cc')})
        ons = {u: int(Zo[f'u{u}_onset'][0]) for u in units}
        tabs = {'mogp': tabs_stats(f'v4_c3_stats_s{sd}.npz', units),
                'proposed': tabs_stats(f'v5c_stats_s{sd}.npz', units)}
        for n in BENCH:
            tabs[n] = tabs_pool(f'v4_bench_resid_s{sd}.npz',
                                f'{n}_resid', units)
        pool = np.concatenate([tabs['mogp'][u][1][tabs['mogp'][u][0] < ons[u]]
                               for u in units])
        sig_ref = pool.std(0) + 1e-12
        for name, T in tabs.items():
            for u in units:
                uc, Tm = T[u]
                Tm = Tm / sig_ref
                h = uc < ons[u]
                fl = float(np.sqrt((Tm[h] ** 2).mean()))
                sg = float(np.abs(Tm[-NEND:]).mean())
                acc.setdefault(name, []).append((fl, sg))
    save = {}
    log(f'\n  {"model":>9} | {"floor":>13} | {"SNR":>13} | cases')
    for name in ['proposed', 'mogp', 'llke', 'bspline', 'lr']:
        a = np.array(acc[name])
        fl, sn = a[:, 0], a[:, 1] / a[:, 0]
        log(f'  {name:>9} | {fl.mean():6.2f} ± {fl.std():4.2f} | '
            f'{sn.mean():6.1f} ± {sn.std():4.1f} | {len(a)}')
        save[f'{name}_floor'] = fl; save[f'{name}_snr'] = sn
    log('  (lr row covers cabn — residual-identical by construction)')
    np.savez(os.path.join(HERE, 'v5c_bench_quality.npz'), **save)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
