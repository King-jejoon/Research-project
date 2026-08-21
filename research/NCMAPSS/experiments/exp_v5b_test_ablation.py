"""
exp_v5b_test_ablation.py — Section-10 ablations evaluated on the sealed
test units = the ELEVENTH DS03 test opening, pre-registered, disclosed
(user decision 2026-08-12: complete the ablation tables on test).

Protocol: every variant is trained on dev only (conditional tables, re-frozen
l1=12 l2=0.25 unless the variant says otherwise); beta by the frozen dev-LOO
NASA rule; test enters as prediction input only.  Cached stats, no GP refits.

Contents:
  (a) HI loss-term variants: base / no_l0 / no_end / no_flat -> test RUL
  (b) raw sensor means replace residuals -> test RUL
      (dev + test tables sampled NPER=200, rng u*7+sd, chain convention)
  (c) armor grid on test: {trim25, mean} x {med3, none} for
      mogp conditional / mogp ungated / llke / bspline / lr / cabn
  (-) healthy-range window: EXCLUDED — its GP was never saved and test
      residuals under that model do not exist in any cache (refit ~1.5 h).
  Gate removal is not recomputed: test numbers already exist from the
  8th and 10th openings (onset 8.54 vs 9.01; RUL 7.52 vs 7.47, p=0.40).
Sanity gates: base must reproduce test 7.52 +/- 0.09; ungated trim25+med3
must reproduce 7.47 +/- 0.06.
Outputs: v5b_test_ablation_results.txt, v5b_test_ablation.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_v5_ablate as A5                    # patched loss (end_w switch)
A5.L1, A5.L2 = 12.0, 0.25
import exp_lib as L
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25, SEEDS
from exp_v4_final import med3

L1, L2 = 12.0, 0.25
VD, VS, NB = 20.2, 19.0, 10
NPER = 200
BENCH = ['llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'v5b_test_ablation_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def summarize(rows, how):
    return trim25(rows) if how == 'trim25' else rows.mean(0)


def nb_of(dur):
    return max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))


# ---------------- table builders (dev + test, gated/ungated/bench) --------
def dev_mogp(sd, how, gated):
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
        V = (VS if nb_of(dur) < NB else VD) if gated else None
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b if V is None else (b & (dc >= V))
            T[i] = summarize(rs[m if m.any() else b], how)
        raw[u] = T; hrs[u] = np.cumsum(dur)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u]-mu)/sg for u in units}, hrs, mu, sg


def test_mogp(sd, how, gated):
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                    if k.endswith('_cc')})
    raw = {}
    for u in units:
        cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']; dc = Zt[f'u{u}_dc']
        ucyc = Zt[f'u{u}_ucyc']; dur = Zt[f'u{u}_hours'].astype(float)
        keep = np.isin(ucyc, np.unique(cc))
        ucyc, dur = ucyc[keep], dur[keep]
        V = (VS if nb_of(dur) < NB else VD) if gated else None
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b if V is None else (b & (dc >= V))
            T[i] = summarize(rs[m if m.any() else b], how)
        raw[u] = T
    return units, raw


def dev_bench(sd, name, how):
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
            T[i] = summarize(rs[m & (cc == c)], how)
        raw[int(u)] = T
        hrs[int(u)] = np.cumsum(hours[int(u)][:len(ucyc)])
    units = sorted(raw)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, {u: (raw[u]-mu)/sg for u in units}, hrs, mu, sg


def test_bench(sd, name, how):
    Zd = np.load(os.path.join(HERE, f'v4_bench_test_resid_s{sd}.npz'))
    cc, uu, rs = Zd['cc'], Zd['uu'], Zd[f'{name}_resid']
    raw = {}
    for u in np.unique(uu):
        m = uu == u
        ucyc = np.unique(cc[m])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = summarize(rs[m & (cc == c)], how)
        raw[int(u)] = T
    return sorted(raw), raw


def raw_sensor_tables(cache):
    """dev + test raw-sensor per-cycle means (chain sampling convention)."""
    SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
    SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
    out = {}
    for split in ['dev', 'test']:
        X = cache[f'X_s_{split}'][:, SIDX]
        A = cache[f'A_{split}']
        unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
        out[split] = (X, unit, cyc)
    return out


# ---------------- evaluation core ----------------------------------------
def run_test(tag, dev_loader, test_loader, variant='base'):
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], []
    for sd in SEEDS:
        units, Zn, hrs, mu, sg = dev_loader(sd)
        np.random.seed(sd)
        if variant == 'no_l0':
            c2 = dict(cfg); c2['lambda0'] = 0.0
            him, _ = train_model_tail([Zn[u] for u in units], **c2)
        elif variant == 'no_flat':
            c2 = dict(cfg); c2['flat_w'] = 0.0
            him, _ = train_model_tail([Zn[u] for u in units], **c2)
        elif variant == 'no_end':
            him = A5.train_patched([Zn[u] for u in units], 0.0, cfg)
        else:
            him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        filt = med3 if variant != 'nomed3' else (lambda h: h)
        HIs = {u: filt(him.forward(Zn[u]).flatten()) for u in units}
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        t_units, t_raw = test_loader(sd)
        Zn_t = {u: (t_raw[u]-mu)/sg for u in t_units}
        HI_t = {u: filt(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, F2, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2-T2)**2).mean())))
        t_ns.append(nasa(P2, T2))
        Ps.append(P2); Ts.append(T2); Fs.append(F2)
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log(f'  {tag:>24}: test {np.mean(t_rm):6.2f} ± {np.std(t_rm):4.2f}  '
        f'NASA {np.mean(t_ns):8.1f}  '
        f'trunc ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    return (float(np.mean(t_rm)), float(np.std(t_rm)), float(np.mean(t_ns)),
            [fr[f] for f in FRACS])


def armor_filtered(tag, how, filt_name):
    """armor cell: variant summary + HI filter, dev-trained, test-scored."""
    def dev_ld(sd, tag=tag, how=how):
        if tag == 'mogp_cond':
            return dev_mogp(sd, how, True)
        if tag == 'mogp_ungated':
            return dev_mogp(sd, how, False)
        return dev_bench(sd, tag, how)

    def test_ld(sd, tag=tag, how=how):
        if tag == 'mogp_cond':
            return test_mogp(sd, how, True)
        if tag == 'mogp_ungated':
            return test_mogp(sd, how, False)
        return test_bench(sd, tag, how)

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    filt = med3 if filt_name == 'med3' else (lambda h: h)
    t_rm = []
    for sd in SEEDS:
        units, Zn, hrs, mu, sg = dev_ld(sd)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: filt(him.forward(Zn[u]).flatten()) for u in units}
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        t_units, t_raw = test_ld(sd)
        Zn_t = {u: (t_raw[u]-mu)/sg for u in t_units}
        HI_t = {u: filt(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, _, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2-T2)**2).mean())))
    return float(np.mean(t_rm)), float(np.std(t_rm))


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B TEST ABLATION — ELEVENTH DS03 opening, pre-registered. '
        'All variants dev-trained (l1=12 conditional unless stated); '
        'test = prediction input only.')
    out = {}

    log('\n(a) HI loss-term variants on test (conditional tables)')
    for v in ['base', 'no_l0', 'no_end', 'no_flat']:
        out[f'loss_{v}'] = run_test(
            v, lambda sd: dev_mogp(sd, 'trim25', True),
            lambda sd: test_mogp(sd, 'trim25', True), variant=v)
    log('  SANITY: base must be 7.52 ± 0.09')

    log('\n(b) raw sensor means on test (NPER=200, rng u*7+sd)')
    cache = L.load_cache()
    tabs = raw_sensor_tables(cache)
    H_test = {sd: np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
              for sd in SEEDS}

    def dev_raw(sd):
        X, unit, cyc = tabs['dev']
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        raw, hrs = {}, {}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cyc_u) if int(c) in pos])
            T = np.empty((len(ucyc), X.shape[1]))
            for i, c in enumerate(ucyc):
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                T[i] = X[r].mean(0)
            raw[int(u)] = T
            hrs[int(u)] = np.cumsum(
                np.array([durs[pos[int(c)]] for c in ucyc], float))
        units = sorted(raw)
        allr = np.concatenate([raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        return units, {u: (raw[u]-mu)/sg for u in units}, hrs, mu, sg

    def test_raw(sd):
        X, unit, cyc = tabs['test']
        raw = {}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ucyc = np.unique(cyc_u)
            T = np.empty((len(ucyc), X.shape[1]))
            for i, c in enumerate(ucyc):
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                T[i] = X[r].mean(0)
            raw[int(u)] = T
        return sorted(raw), raw

    out['raw'] = run_test('raw sensor means', dev_raw, test_raw)

    log('\n(c) armor grid on test: {trim25, mean} x {med3, none}')
    for tag in ['mogp_cond', 'mogp_ungated'] + BENCH:
        vals = []
        for how in ['trim25', 'mean']:
            for fn in ['med3', 'none']:
                r = armor_filtered(tag, how, fn)
                out[f'armor_{tag}_{how}_{fn}'] = r
                vals.append(f'{how}+{fn} {r[0]:.2f}±{r[1]:.2f}')
        log(f'  {tag:>13}: ' + ' | '.join(vals))
    log('  SANITY: mogp_cond trim25+med3 = 7.52, mogp_ungated = 7.47')

    log('\n(-) healthy-range window: EXCLUDED (GP never saved; test '
        'residuals require ~1.5 h refit).  Gate removal: already on test '
        '(8th/10th openings): onset 8.54 vs 9.01; RUL 7.52 vs 7.47 p=0.40.')
    np.savez(os.path.join(HERE, 'v5b_test_ablation.npz'),
             **{k: np.array(v[:3] if len(v) > 3 else v, dtype=object)
                for k, v in out.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
