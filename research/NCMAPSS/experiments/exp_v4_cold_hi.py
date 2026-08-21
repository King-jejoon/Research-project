"""
exp_v4_cold_hi.py — cold-side channel discriminator: the frozen HI + RUL
recipe fed ONLY the cold-side channels [T30, Nc, Wf] (T48/T50 removed).

Rationale (diagnostic batch): the truncation-RUL end metric is insensitive
to the normal model because every model captures the hot-section drift in
T48/T50 and the HI fusion puts 70-79 % of its weight there.  Removing the
hot channels asks each residual stream to survive on the channels where the
models actually differ (experiment 1: LR/CaBN bury T30/Nc/Wf in noise,
B-spline loses Wf).  DEV ONLY — LOO truncation RUL; test is not touched.

Per seed and model: dev full-life trim25 tables restricted to the 3 cold
columns -> pooled dev z-norm -> HI (l1=8, l2=0.25, med3, frozen recipe,
input_dim=3) -> beta dev-LOO NASA -> LOO truncation RUL.
mogp rows: gated V*=20.2 and ungated, from the chain's cached stats.
Competitor rows: cached benchmark residuals (v4_bench_resid_s{sd}.npz).
Outputs: v4_cold_hi_results.txt, v4_cold_hi.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, \
    train_model_tail, shape_ok
from exp_v4_hi import build_tables, trim25, SEEDS, VSTAR
from exp_v4_final import med3

COLD = [0, 3, 4]                  # T30, Nc, Wf within [T30,T48,T50,Nc,Wf]
L1, L2 = 8.0, 0.25
VOFF = -1e9
RES = os.path.join(HERE, 'v4_cold_hi_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def mogp_tables(sd, V):
    """chain trim25 tables at gate V, restricted to the cold columns."""
    units, Zn, hrs = build_tables(sd, V)   # Zn already pooled-z per channel
    return units, {u: Zn[u][:, COLD] for u in units}, hrs


def bench_tables(sd, name):
    """competitor trim25 tables (ungated, as in the benchmark), cold cols."""
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
    Zn = {u: ((raw[u] - mu) / sg)[:, COLD] for u in units}
    return units, Zn, hrs


def run(tag, loader):
    rms, nss, oks = [], [], []
    for sd in SEEDS:
        units, Zn, hrs = loader(sd)
        np.random.seed(sd)
        cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2,
                                       end_target=1.03)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, (st_, en_, _) = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok))
        log(f'  {tag} seed{sd}: shape={"OK" if ok else "FAIL"} '
            f'(start {st_:.2f} end {en_:.2f}) beta={b} '
            f'LOO RMSE={rms[-1]:.2f} NASA={nss[-1]:.1f}')
    return (float(np.mean(rms)), float(np.std(rms)),
            float(np.mean(nss)), int(np.sum(oks)))


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('COLD-SIDE CHANNEL DISCRIMINATOR — HI on [T30, Nc, Wf] only, '
        'frozen recipe, dev LOO (test untouched)')
    R = {}
    R['mogp gated'] = run('mogp gated', lambda sd: mogp_tables(sd, VSTAR))
    R['mogp ungated'] = run('mogp ungated', lambda sd: mogp_tables(sd, VOFF))
    for name in ['llke', 'bspline', 'lr', 'cabn']:
        R[name] = run(name, lambda sd, n=name: bench_tables(sd, n))
    log('')
    log('SUMMARY — dev LOO truncation RUL, cold channels only')
    log(f'  {"model":>13} | {"RUL RMSE":>13} | {"NASA":>8} | shape')
    for k, (m, s, n, sh) in R.items():
        log(f'  {k:>13} | {m:6.2f} ± {s:4.2f} | {n:8.1f} | {sh}/3')
    np.savez(os.path.join(HERE, 'v4_cold_hi.npz'),
             **{k.replace(' ', '_'): np.array(v) for k, v in R.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
