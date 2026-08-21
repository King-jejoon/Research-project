"""
exp_v4_ablate2raw.py — ablation 2: replace the MOGP residuals with RAW
sensor per-cycle means as the HI input.  Everything downstream identical
(pooled dev z-normalization, HI l1=8 l2=0.25 med3, beta dev-LOO, dev LOO
truncation RUL).  No gate (no detcov exists without a model).
Tests whether the conditional-model residual is necessary at all.
Output: v4_ablate2raw_results.txt
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_final import med3

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
SEEDS = [0, 1, 2]
NPER = 200
L1, L2 = 8.0, 0.25
RES = os.path.join(HERE, 'v4_ablate2raw_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('ABLATION 2 — raw sensor cycle means replace the MOGP residuals '
        '(downstream frozen; no gate possible)')
    cache = L.load_cache()
    X, A = cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    rms, nss, oks = [], [], []
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        raw, hrs = {}, {}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cyc_u) if int(c) in pos])
            T = np.empty((len(ucyc), len(SENS)))
            for i, c in enumerate(ucyc):
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                T[i] = X[r].mean(0)                 # raw per-cycle mean
            raw[int(u)] = T
            hrs[int(u)] = np.cumsum(
                np.array([durs[pos[int(c)]] for c in ucyc], float))
        units = sorted(raw)
        allr = np.concatenate([raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn = {u: (raw[u] - mu) / sg for u in units}
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, (st_, en_, mo_) = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok))
        log(f'seed{sd}: shape={"OK" if ok else "FAIL"} (start {st_:.2f} '
            f'end {en_:.2f}) beta={b} LOO RUL={rms[-1]:.2f} NASA={nss[-1]:.1f}')
    log('')
    log(f'RAW-MEAN input : RUL RMSE={np.mean(rms):.2f} ± {np.std(rms):.2f}  '
        f'NASA={np.mean(nss):.1f}  shape {int(np.sum(oks))}/3')
    log(f'chain reference: RUL RMSE=7.30 ± 0.16  NASA=27.4  shape 3/3')
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
