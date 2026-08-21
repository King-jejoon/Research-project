"""
exp_v5b_armor_cold_bench.py — three dev-only re-computations on the
RE-FROZEN chain (lambda1 = 12, lambda2 = 0.25, conditional gate):

A. Armor-removal grid, MOGP conditional row: {trim25, mean} x {med3, none}.
B. Cold-side channel stress test [T30, Nc, Wf]: MOGP conditional / ungated
   + competitors (protocol = exp_v4_cold_hi, l12).
C. Benchmark dev LOO ranking, all models, l12 downstream (component-swap).
Dev only, cached stats, no test contact.
Outputs: v5b_acb_results.txt, v5b_acb.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_hi import build_tables, trim25, SEEDS
from exp_v4_final import med3
from exp_v5_armor import cond_tables, summarize

L1, L2 = 12.0, 0.25
VOFF = -1e9
COLD = [0, 3, 4]                       # T30, Nc, Wf in [T30,T48,T50,Nc,Wf]
BENCH = ['llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'v5b_acb_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def bench_tables(sd, name, cold=False):
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
            T[i] = trim25(rs[m & (cc == c)])
        raw[int(u)] = T
        hrs[int(u)] = np.cumsum(hours[int(u)][:len(ucyc)])
    units = sorted(raw)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Zn = {u: (raw[u] - mu) / sg for u in units}
    if cold:
        Zn = {u: Zn[u][:, COLD] for u in units}
    return units, Zn, hrs


def run_loo(loader, filt='med3'):
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    rms, nss, oks, errs = [], [], [], []
    for sd in SEEDS:
        units, Zn, hrs = loader(sd)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: him.forward(Zn[u]).flatten() for u in units}
        if filt == 'med3':
            HIs = {u: med3(h) for u, h in HIs.items()}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok)); errs.append(P - T)
    return (float(np.mean(rms)), float(np.std(rms)), float(np.mean(nss)),
            int(np.sum(oks)), np.concatenate(errs))


def cond_loader(how):
    def f(sd):
        units, Zn, hrs = cond_tables(sd, how)
        return units, Zn, hrs
    return f


def cond_cold(sd):
    units, Zn, hrs = cond_tables(sd, 'trim25')
    return units, {u: Zn[u][:, COLD] for u in units}, hrs


def ungated_cold(sd):
    units, Zn, hrs = build_tables(sd, VOFF)
    return units, {u: Zn[u][:, COLD] for u in units}, hrs


def main():
    open(RES, 'w').close()
    t0 = time.time()
    out = {}
    log('V5B ARMOR/COLD/BENCH — re-frozen l1=12, l2=0.25, dev only')

    log('\nA. armor grid, MOGP conditional row')
    for how in ['trim25', 'mean']:
        for filt in ['med3', 'none']:
            r = run_loo(cond_loader(how), filt)
            out[f'armor_{how}_{filt}'] = r[:4]
            log(f'  {how:>6}+{filt:<5}: RUL {r[0]:.2f} ± {r[1]:.2f}  '
                f'shape {r[3]}/3')

    log('\nB. cold-side [T30, Nc, Wf], l12 recipe')
    for tag, ld in ([('mogp cond', cond_cold), ('mogp ungated', ungated_cold)]
                    + [(n, (lambda s, n=n: bench_tables(s, n, cold=True)))
                       for n in BENCH]):
        r = run_loo(ld)
        out[f'cold_{tag.replace(" ", "_")}'] = r[:4]
        log(f'  {tag:>12}: RUL {r[0]:.2f} ± {r[1]:.2f}  NASA {r[2]:.1f}  '
            f'shape {r[3]}/3')

    log('\nC. benchmark dev LOO ranking, l12 downstream (5 channels)')
    for tag, ld in ([('mogp cond', cond_loader('trim25')),
                     ('mogp ungated',
                      lambda sd: build_tables(sd, VOFF))]
                    + [(n, (lambda s, n=n: bench_tables(s, n)))
                       for n in BENCH]):
        r = run_loo(ld)
        out[f'bench_{tag.replace(" ", "_")}'] = r[:4]
        log(f'  {tag:>12}: RUL {r[0]:.2f} ± {r[1]:.2f}  NASA {r[2]:.1f}  '
            f'shape {r[3]}/3')

    np.savez(os.path.join(HERE, 'v5b_acb.npz'),
             **{k: np.array(v) for k, v in out.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
