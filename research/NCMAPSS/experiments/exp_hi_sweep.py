"""
exp_hi_sweep.py — HI physics-loss coefficient sweep on the FINAL config
(3 sensors, matern32 r1, NPER=200, drop 60%), reusing cached per-point stats
so no GP recomputation is needed.

Loss (neural_fusion): L = l_init*term0 + 1.0*term1 + l_mono*term2 + l_conv*term3
  term0: initial HI <= init_thr   (coef l_init,  current 1.0)
  term1: last HI = 1              (coef fixed 1.0 = reference scale)
  term2: monotonicity             (coef l_mono,  current 6.0)
  term3: convexity                (coef l_conv,  current 2.0)
One-at-a-time sweep around the current values + init_thr bonus sweep.

Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_sweep.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_ds03_port import rul_eval, nasa_score, FRACS
import neural_fusion as NF

SENS = ['T48', 'T50', 'Wf']
DROP = 0.60
SEEDS = [0, 1, 2]
CACHE = os.path.join(HERE, 'port_stats_s3_matern32_r1_m18_n8192_p200_s{sd}.npz')
RES = os.path.join(HERE, 'hi_sweep_results.txt')

BASE = dict(l_init=1.0, l_mono=6.0, l_conv=2.0, thr=0.2)
# stage 2: combine the individually-best directions + extend lower
CONFIGS = [
    ('base(6,2)', dict(BASE)),
    ('mono2+conv0.5', dict(BASE, l_mono=2.0, l_conv=0.5)),
    ('mono1+conv0.5', dict(BASE, l_mono=1.0, l_conv=0.5)),
    ('mono2+conv0.25', dict(BASE, l_mono=2.0, l_conv=0.25)),
    ('mono1+conv0.25', dict(BASE, l_mono=1.0, l_conv=0.25)),
    ('mono0.5+conv0.25', dict(BASE, l_mono=0.5, l_conv=0.25)),
    ('combo+init4', dict(BASE, l_mono=2.0, l_conv=0.5, l_init=4.0)),
    ('combo+thr0.1', dict(BASE, l_mono=2.0, l_conv=0.5, thr=0.1)),
    ('combo+all', dict(BASE, l_mono=2.0, l_conv=0.5, l_init=4.0, thr=0.1)),
]


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def build_residuals(sd):
    """Filtered (drop 60%) per-cycle z-normalized residual sequences from cache."""
    z = np.load(CACHE.format(sd=sd), allow_pickle=False)
    du = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('dev_') and k.endswith('_cc')})
    tu = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('test_') and k.endswith('_cc')})
    def resid_dict(units, split):
        out = {}
        for u in units:
            key = f'{split}_{u}'
            cc = z[f'{key}_cc']; r = z[f'{key}_resid']; dc = z[f'{key}_detcov']
            thr = np.percentile(dc, DROP * 100)
            keep = dc >= thr
            ucyc = np.unique(cc)
            d = {s: {} for s in SENS}
            for k in ucyc:
                mk = (cc == k) & keep
                if mk.sum() == 0: mk = cc == k
                mv = r[mk].mean(0)
                for j, s in enumerate(SENS): d[s][int(k)] = float(mv[j])
            out[u] = d
        return out
    dev_res = resid_dict(du, 'dev'); test_res = resid_dict(tu, 'test')
    st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
              np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in SENS}
    nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
    dev_n, test_n = nrm(dev_res), nrm(test_res)
    def blist(dn):
        out = []
        for u in sorted(dn):
            cy = sorted(dn[u][SENS[0]].keys())
            out.append((u, cy, np.array([[dn[u][s][k] for s in SENS] for k in cy])))
        return out
    return blist(dev_n), blist(test_n)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'HI COEFFICIENT SWEEP  (final config: 3sens/m32r1/NPER200/drop60)  seeds={SEEDS}')
    log(f'base: l_init=1.0  l_end=1.0(fixed ref)  l_mono=6.0  l_conv=2.0  init_thr=0.2')

    # residuals are lambda-independent -> build once per seed
    RD = {sd: build_residuals(sd) for sd in SEEDS}
    log(f'residuals ready ({time.time()-t0:.0f}s)')
    log(f'\n{"config":>14} | {"RMSE s0/s1/s2":>20} | {"mean":>6} | {"std":>5} | {"score":>6}')
    log('-' * 68)
    for name, cfg in CONFIGS:
        rmses, scores = [], []
        for sd in SEEDS:
            dl, tl = RD[sd]
            np.random.seed(sd)
            him, _ = NF.train_model([d for _, _, d in dl], epochs=1000,
                                    lambda0=cfg['l_init'], lambda1=cfg['l_mono'], lambda2=cfg['l_conv'],
                                    init_threshold=cfg['thr'], alpha=0.001, verbose=False)
            dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
            tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
            P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
            rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
            scores.append(nasa_score(P, T))
        rmses = np.array(rmses); scores = np.array(scores)
        log(f'{name:>14} | {rmses[0]:>6.2f} {rmses[1]:>6.2f} {rmses[2]:>6.2f} | '
            f'{rmses.mean():>6.2f} | {rmses.std():>5.2f} | {scores.mean():>6.1f}')
    log(f'\nwall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
