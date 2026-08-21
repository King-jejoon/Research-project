"""
exp_v5b_dev.py — chain RE-FREEZE on the conditional-table lambda re-sweep
winner (user decision 2026-08-10): lambda1 = 12, lambda2 = 0.25 (property
composite rank 1, shape 3/3; the previous recipe (8, 0.25) ranks third).

Official dev derivation of the re-frozen chain — conditional gate tables,
lambda12 HI + med3, beta by the frozen dev-LOO NASA rule.  Also recomputes
the lambda8 chain on the same tables for a paired dev comparison.
Dev only, cached stats, no test contact.
Outputs: v5b_dev_results.txt, v5b_dev.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, train_model_tail, \
    shape_ok
from exp_v4_final import med3
from exp_v5_full_rul import dev_tables
from exp_v4_hi import SEEDS

RES = os.path.join(HERE, 'v5b_dev_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def run_chain(l1, l2, data):
    cfg = dict(HI_CFG); cfg.update(lambda1=l1, lambda2=l2, end_target=1.03)
    rms, nss, oks, bs, Ps, Ts, Fs = [], [], [], [], [], [], []
    for sd in SEEDS:
        units, raw, Zn, hrs, mu, sg = data[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok)); bs.append(b)
        Ps.append(P); Ts.append(T); Fs.append(F)
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    return dict(rms=rms, nss=nss, oks=oks, bs=bs, P=P, T=T, F=F, fr=fr)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B DEV — re-frozen chain lambda1=12, lambda2=0.25 '
        '(conditional gate everywhere), vs lambda8 on identical tables')
    data = {sd: dev_tables(sd) for sd in SEEDS}
    out = {}
    for tag, l1 in [('l12', 12.0), ('l8', 8.0)]:
        r = run_chain(l1, 0.25, data)
        out[tag] = r
        log(f'{tag}: dev LOO = {np.mean(r["rms"]):.2f} ± '
            f'{np.std(r["rms"]):.2f}  NASA {np.mean(r["nss"]):.1f} ± '
            f'{np.std(r["nss"]):.1f}  shape {int(np.sum(r["oks"]))}/3  '
            f'beta={r["bs"]}')
        log('  per-trunc = ' + ' / '.join(f'{r["fr"][f]:.2f}' for f in FRACS))
    pw = st.wilcoxon(np.abs(out['l12']['P'] - out['l12']['T']),
                     np.abs(out['l8']['P'] - out['l8']['T']),
                     zero_method='zsplit').pvalue
    log(f'\npaired |err| lambda12 vs lambda8 (108 preds): p={pw:.3f}')
    np.savez(os.path.join(HERE, 'v5b_dev.npz'),
             **{f'{tag}_{k}': np.asarray(v) for tag in out
                for k, v in out[tag].items() if k in ('P', 'T', 'F', 'rms',
                                                      'nss', 'bs')})
    log(f'wall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
