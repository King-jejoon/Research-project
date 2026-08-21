"""
exp_v4_ablate13.py — sensitivity experiments 1 and 3 on the FINAL frozen
chain only (mogp set5 rbf r1, V*=20.2, HI l1=8 l2=0.25, med3, beta dev-LOO).
All dev; test untouched.

Experiment 1 — gate ablation, scored on the FINAL metric:
  gate off + absolute V sweep 18.1..20.5 -> dev LOO truncation RUL RMSE.

Experiment 3 — frozen HI loss-term ablation (delete one at a time):
  base / -lambda0 (initial level) / -end (end-level anchor, source-patched
  end_w switch) / -flat (tail-flatness term).  Scored by dev LOO RUL RMSE,
  NASA and the shape gate.
Outputs: v4_ablate13_results.txt, v4_ablate13.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_hi import build_tables, SEEDS, VSTAR
from exp_v4_final import med3
from neural_fusion import NeuralDataFusionModel, AdamOptimizer

L1, L2 = 8.0, 0.25
RES = os.path.join(HERE, 'v4_ablate13_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


# ---- patched loss with an end-term switch (end_w) ----
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
exec(compile(_src, 'neural_fusion_tail_patched', 'exec'), _ns)
_loss_patched = _ns['compute_loss_and_gradients_tail']
assert 'end_w=1.0' in _src


def train_patched(data, end_w=1.0, **cfg):
    model = NeuralDataFusionModel(input_dim=data[0].shape[1])
    opt = AdamOptimizer(model.parameters, cfg.get('alpha', 0.001))
    for _ in range(cfg.get('epochs', 1000)):
        _loss_patched(model, data,
                      lambda0=cfg.get('lambda0', 1.0),
                      lambda1=cfg.get('lambda1', L1),
                      lambda2=cfg.get('lambda2', L2),
                      init_threshold=cfg.get('init_threshold', 0.2),
                      end_target=cfg.get('end_target', 1.03),
                      flat_w=cfg.get('flat_w', 800.0),
                      flat_m=cfg.get('flat_m', 0.004),
                      end_w=end_w)
        opt.step(model.parameters, model.gradients)
    return model


def chain_rul(sd, V, variant='base'):
    """trim25 at gate V -> HI (variant) -> med3 -> dev LOO RUL."""
    units, Zn, hrs = build_tables(sd, V)
    np.random.seed(sd)
    kw = dict(HI_CFG); kw.update(lambda1=L1, lambda2=L2, end_target=1.03)
    if variant == 'base':
        him, _ = train_model_tail([Zn[u] for u in units], **kw)
    elif variant == 'no_l0':
        kw['lambda0'] = 0.0
        him, _ = train_model_tail([Zn[u] for u in units], **kw)
    elif variant == 'no_flat':
        kw['flat_w'] = 0.0
        him, _ = train_model_tail([Zn[u] for u in units], **kw)
    elif variant == 'no_end':
        him = train_patched([Zn[u] for u in units], end_w=0.0,
                            lambda0=kw['lambda0'], lambda1=L1, lambda2=L2,
                            init_threshold=kw['init_threshold'],
                            end_target=kw['end_target'],
                            flat_w=kw['flat_w'], flat_m=kw['flat_m'],
                            epochs=kw['epochs'], alpha=kw['alpha'])
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    ok, (st_, en_, mo_) = shape_ok(HIs)
    P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
    return (float(np.sqrt(((P - T) ** 2).mean())), nasa(P, T), int(ok),
            float(st_), float(en_), P - T)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('SENSITIVITY 1 — gate ablation scored by dev LOO RUL RMSE '
        '(final chain, HI recipe frozen)')
    grid = [None] + list(np.round(np.arange(18.1, 20.51, 0.1), 1))
    out1 = {}
    err_off = None
    for V in grid:
        rms, nss, errs = [], [], []
        for sd in SEEDS:
            r, n_, ok, st_, en_, e = chain_rul(sd, -1e9 if V is None else V,
                                               'base')
            rms.append(r); nss.append(n_); errs.append(e)
        e_all = np.concatenate(errs)
        key = 'off' if V is None else f'{V:.1f}'
        out1[key] = (float(np.mean(rms)), float(np.std(rms)),
                     float(np.mean(nss)))
        if V is None:
            err_off = e_all
            log(f'  gate off : RUL RMSE={np.mean(rms):.2f} ± {np.std(rms):.2f}'
                f'  NASA={np.mean(nss):.1f}')
        else:
            pw = st.wilcoxon(np.abs(e_all), np.abs(err_off),
                             zero_method='zsplit').pvalue
            log(f'  V={V:4.1f} : RUL RMSE={np.mean(rms):.2f} ± '
                f'{np.std(rms):.2f}  NASA={np.mean(nss):.1f}  p={pw:.3f}')
    np.savez(os.path.join(HERE, 'v4_ablate1.npz'),
             **{k: np.array(v) for k, v in out1.items()})

    log('')
    log('SENSITIVITY 3 — HI loss-term ablation at V*=20.2 '
        '(delete one at a time)')
    log(f'  {"variant":>8} | {"RUL RMSE":>13} | {"NASA":>11} | '
        f'{"shape":>5} | start/end')
    out3 = {}
    for variant in ['base', 'no_l0', 'no_end', 'no_flat']:
        rms, nss, oks, sts, ens = [], [], [], [], []
        for sd in SEEDS:
            r, n_, ok, st_, en_, _ = chain_rul(sd, VSTAR, variant)
            rms.append(r); nss.append(n_); oks.append(ok)
            sts.append(st_); ens.append(en_)
        out3[variant] = (float(np.mean(rms)), float(np.std(rms)),
                         float(np.mean(nss)), int(np.sum(oks)))
        log(f'  {variant:>8} | {np.mean(rms):5.2f} ± {np.std(rms):4.2f} | '
            f'{np.mean(nss):5.1f} ± {np.std(nss):3.1f} | {int(np.sum(oks))}/3'
            f' | {np.mean(sts):.2f}/{np.mean(ens):.2f}')
    np.savez(os.path.join(HERE, 'v4_ablate3.npz'),
             **{k: np.array(v) for k, v in out3.items()})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
