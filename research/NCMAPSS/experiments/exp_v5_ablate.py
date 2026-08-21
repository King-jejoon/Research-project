"""
exp_v5_ablate.py — ablation suite re-baselined on the FINAL v5 chain
(coverage-conditional gate everywhere).  Dev only, cached stats.

1. HI loss-term ablation on conditional tables: base / no_l0 / no_end /
   no_flat (end term via the source-patched end_w switch, as v4_ablate13).
2. Gate-vs-ungated on the final metric: conditional vs ungated dev LOO
   RUL with paired p; conditional vs ungated dev onset with paired p.
Outputs: v5_ablate_results.txt
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_hi import trim25, SEEDS
from exp_v4_final import med3
from exp_v5_full_rul import dev_tables            # conditional per-unit V
from exp_v4_gate_redesign import unit_data, curve, detect_var, LONG
from neural_fusion import NeuralDataFusionModel, AdamOptimizer

L1, L2 = 8.0, 0.25
RES = os.path.join(HERE, 'v5_ablate_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


# ---- patched loss with an end-term switch (as in exp_v4_ablate13) ----
_src = open(os.path.join(HERE, 'neural_fusion_tail.py'), encoding='utf-8').read()
_src = _src.replace(
    "ceil_w=0.0, cmax=0.92, ceil_frac=0.90, end_target=1.0,\n"
    "                                    flat_w=0.0, flat_m=0.0, flat2_w=0.0, flat2_m=0.0):",
    "ceil_w=0.0, cmax=0.92, ceil_frac=0.90, end_target=1.0,\n"
    "                                    flat_w=0.0, flat_m=0.0, flat2_w=0.0, flat2_m=0.0,\n"
    "                                    end_w=1.0):")
_src = _src.replace("loss_term1 += (HI_n[-1, 0] - end_target) ** 2",
                    "loss_term1 += end_w * (HI_n[-1, 0] - end_target) ** 2")
_src = _src.replace("grad_h[-1, 0] += 2 * (HI_n[-1, 0] - end_target)",
                    "grad_h[-1, 0] += end_w * 2 * (HI_n[-1, 0] - end_target)")
_ns = {}
exec(compile(_src, 'nft_patched', 'exec'), _ns)
_loss_p = _ns['compute_loss_and_gradients_tail']
assert 'end_w=1.0' in _src


def train_patched(data, end_w, cfg):
    model = NeuralDataFusionModel(input_dim=data[0].shape[1])
    opt = AdamOptimizer(model.parameters, cfg.get('alpha', 0.001))
    for _ in range(cfg.get('epochs', 1000)):
        _loss_p(model, data, lambda0=cfg.get('lambda0', 1.0), lambda1=L1,
                lambda2=L2, init_threshold=cfg.get('init_threshold', 0.2),
                end_target=cfg.get('end_target', 1.03),
                flat_w=cfg.get('flat_w', 800.0),
                flat_m=cfg.get('flat_m', 0.004), end_w=end_w)
        opt.step(model.parameters, model.gradients)
    return model


def rul_variant(sd, variant, gated=True):
    if gated:
        units, raw, Zn, hrs, mu, sg = dev_tables(sd)
    else:
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
                T[i] = trim25(rs[cc == c])
            raw[u] = T
            hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
        allr = np.concatenate([raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn = {u: (raw[u]-mu)/sg for u in units}
    np.random.seed(sd)
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    if variant == 'base':
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    elif variant == 'no_l0':
        cfg['lambda0'] = 0.0
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    elif variant == 'no_flat':
        cfg['flat_w'] = 0.0
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    elif variant == 'no_end':
        him = train_patched([Zn[u] for u in units], 0.0, cfg)
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    ok, (st_, en_, _) = shape_ok(HIs)
    P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
    return (float(np.sqrt(((P-T)**2).mean())), nasa(P, T), int(ok),
            float(st_), float(en_), P - T)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5 ABLATIONS — re-baselined on the coverage-conditional chain, dev only')
    log('\n1. HI loss-term ablation (conditional tables)')
    log(f'  {"variant":>8} | {"RUL RMSE":>13} | {"NASA":>8} | shape | start/end')
    for v in ['base', 'no_l0', 'no_end', 'no_flat']:
        rms, nss, oks, sts, ens = [], [], [], [], []
        for sd in SEEDS:
            r, n_, ok, s_, e_, _ = rul_variant(sd, v)
            rms.append(r); nss.append(n_); oks.append(ok)
            sts.append(s_); ens.append(e_)
        log(f'  {v:>8} | {np.mean(rms):6.2f} ± {np.std(rms):4.2f} | '
            f'{np.mean(nss):8.1f} | {int(np.sum(oks))}/3 | '
            f'{np.mean(sts):.2f}/{np.mean(ens):.2f}')

    log('\n2. Gate vs ungated on the final metric (conditional chain)')
    eg, eu = [], []
    for sd in SEEDS:
        _, _, _, _, _, e1 = rul_variant(sd, 'base', gated=True)
        _, _, _, _, _, e0 = rul_variant(sd, 'base', gated=False)
        eg.append(e1); eu.append(e0)
    eg = np.concatenate(eg); eu = np.concatenate(eu)
    pw = st.wilcoxon(np.abs(eg), np.abs(eu), zero_method='zsplit').pvalue
    log(f'  RUL: conditional {np.sqrt((eg**2).mean()):.2f} vs ungated '
        f'{np.sqrt((eu**2).mean()):.2f}, paired p={pw:.3f}')

    D = unit_data()
    e_c = np.array([detect_var(curve(d, 19.0 if d['nbase'] < 10 else 20.2),
                               d['dur'], d['ucyc'], 'std') - d['true']
                    for d in D], float)
    e_u = np.array([detect_var(curve(d, None), d['dur'], d['ucyc'], 'std')
                    - d['true'] for d in D], float)
    po = st.wilcoxon(np.abs(e_c), np.abs(e_u), zero_method='zsplit').pvalue
    log(f'  onset: conditional {np.sqrt((e_c**2).mean()):.2f} vs ungated '
        f'{np.sqrt((e_u**2).mean()):.2f}, paired p={po:.3f}')
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
