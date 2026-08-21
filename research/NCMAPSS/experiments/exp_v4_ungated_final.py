"""
exp_v4_ungated_final.py — the frozen v4 chain WITHOUT the detcov gate,
rendered on dev (LOO) and on the already-opened DS03 test stats.

Purpose: document the ungated MOGP row of the benchmark (diagnostic batch
established dev 7.12 +/- 0.06, test 7.47 +/- 0.05; this run re-derives the
truncation breakdown from the SAME cached statistics so the benchmark table
carries no blank cells).  No new GP fits and no new test opening: every
number below is a re-summary of v4_c3_stats_s{sd}.npz (dev) and
v4_test_stats_s{sd}.npz (test), the exact caches behind the official
gated verification (exp_v4_final).

Chain identical to exp_v4_final except V = -inf everywhere (gate off):
trim25 of ALL points -> dev pooled z-norm -> HI l1=8 l2=0.25 med3 ->
beta dev-LOO NASA -> truncation RUL 20/40/60/80 %.
Outputs: v4_ungated_final_results.txt, v4_ungated_final.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import build_tables, SEEDS
from exp_v4_test import summaries
from exp_v4_final import med3

L1, L2 = 8.0, 0.25
VOFF = -1e9                       # gate off
RES = os.path.join(HERE, 'v4_ungated_final_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V4 UNGATED CHAIN (gate off, everything else frozen) — dev LOO + '
        'test re-summary from the official cached stats')
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)

    dev_rm, dev_ns, betas = [], [], []
    dev_frac = {f: [] for f in FRACS}
    t_rm, t_ns = [], []
    t_allP, t_allT, t_allF = [], [], []
    for sd in SEEDS:
        units, Zn, hrs = build_tables(sd, VOFF)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, (st_, en_, mo_) = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        dev_rm.append(float(np.sqrt(((P - T) ** 2).mean())))
        dev_ns.append(nasa(P, T)); betas.append(b)
        for f in FRACS:
            m = F == f
            dev_frac[f].append(float(np.sqrt(((P[m] - T[m]) ** 2).mean())))
        log(f'seed{sd} dev: shape={"OK" if ok else "FAIL"} (start {st_:.2f} '
            f'end {en_:.2f})  LOO RMSE={dev_rm[-1]:.2f}  NASA={dev_ns[-1]:.1f}'
            f'  beta={b}')

        # ---- test re-summary (cached stats; ungated trim25) ----
        Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
        t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                          if k.endswith('_cc')})
        t_raw, _, _, _, _, _ = summaries(Zt, VOFF, t_units)
        # dev pooled normalization of the UNGATED tables, identical rule
        from exp_v4_hi import trim25
        Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        dev_raw = {}
        for u in units:
            cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']
            ur = H[f'dev_{u}_ucyc']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            Tm = np.empty((len(ucyc), rs.shape[1]))
            for i, c in enumerate(ucyc):
                Tm[i] = trim25(rs[cc == c])
            dev_raw[u] = Tm
        allr = np.concatenate([dev_raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
        HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P, T, F, _ = eval_units(betas[-1], cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P - T) ** 2).mean())))
        t_ns.append(nasa(P, T))
        t_allP.append(P); t_allT.append(T); t_allF.append(F)
        log(f'seed{sd} test: RMSE={t_rm[-1]:.2f}  NASA={t_ns[-1]:.1f}')

    log('')
    log('UNGATED chain summary (3 seeds):')
    log(f'  dev LOO RUL  = {np.mean(dev_rm):.2f} ± {np.std(dev_rm):.2f}  '
        f'NASA {np.mean(dev_ns):.1f} ± {np.std(dev_ns):.1f}  beta={betas}')
    log(f'  dev per-trunc = '
        + ' / '.join(f'{np.mean(dev_frac[f]):.2f}' for f in FRACS))
    P = np.concatenate(t_allP); T = np.concatenate(t_allT)
    F = np.concatenate(t_allF)
    log(f'  test RUL     = {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}')
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log(f'  test per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    # paired check vs the official gated predictions
    Zm = np.load(os.path.join(HERE, 'v4_final.npz'))
    eg = np.abs(Zm['P'] - Zm['T']); eu = np.abs(P - T)
    if eg.shape == eu.shape:
        pw = st.wilcoxon(eu, eg, zero_method='zsplit').pvalue
        log(f'  paired |err| Wilcoxon ungated vs gated (72 preds): p={pw:.3f}')
    np.savez(os.path.join(HERE, 'v4_ungated_final.npz'),
             P=P, T=T, F=F, dev_rm=np.array(dev_rm), t_rm=np.array(t_rm),
             dev_ns=np.array(dev_ns), t_ns=np.array(t_ns),
             frac=np.array([fr[f] for f in FRACS]),
             dev_frac=np.array([np.mean(dev_frac[f]) for f in FRACS]))
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
