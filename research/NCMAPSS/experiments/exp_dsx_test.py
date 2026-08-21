"""
exp_dsx_test.py — transfer chain (v6), PHASE 3: PRE-REGISTERED one-time
test batch = the FIRST v6 test opening of this subset, disclosed.
(Disclosure: DS01/DS02 test splits were also read by the retired July-2026
5-stage pipeline; the v6 ledger starts at 1 for every subset.)

Everything below is frozen on dev BEFORE this run (Phases 1-2):
  detector rule ({DS}v6_detsel.npz), HI #1 recipe ({DS}v6_hi1.npz),
  HI #2 recipe ({DS}v6_stage2.npz, tie-band rule), stage-1/2 GPs (.pt).
Pre-registration:
  (a) ONSET — conditional gate vs ungated on the stage-1 GP, test units:
      RMSE overall and per nbase group (sparse / dense), paired Wilcoxon
      over units x seeds; dev recap from Phase 1b.
  (b) RUL — HI #2 chain (healthy-range GP) and HI #1 chain (cycle<=3 GP):
      truncation RUL 20/40/60/80 %, overall RMSE, NASA (sum per seed, mean
      over seeds), per-level RMSE + score; paired Wilcoxon HI#2 vs HI#1.
  (c) BENCHMARK — llke (h=0.1) / bspline (30 knots) / lr / cabn (lam1=0),
      hyperparameters frozen at their DS03 CV values, trained per seed on
      the identical cycle<=3 rows of the stage-1 GP, HI #1 recipe
      downstream retrained per model (ungated), dev LOO -> beta -> test;
      paired Wilcoxon vs the HI #2 chain.
  (d) every number reported regardless of direction; unit n caveat.
Sanity gates: dev HI #1 / HI #2 LOO must reproduce Phases 1c/2 within 0.02
before any test array is read.
Usage: DS=... python3 exp_dsx_test.py
Outputs: {DS}v6_test_results.txt, {DS}v6_c3_test_stats_s{sd}.npz,
         {DS}v6_hr_test_stats_s{sd}.npz, {DS}v6_bench_{dev,test}_resid_s{sd}.npz,
         {DS}v6_test.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_bench2_lib as B2
from exp_v4_c3_detcov import SIDX
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3
from exp_dsx_lib import load_stats, q75_curve, detect_p, gate_V, \
    nbase_at, tables_ungated, point_stats_robust, point_stats_parallel, stats_parallel
SEEDS = [int(x) for x in os.environ.get('SEEDS_ONLY', '0,1,2').split(',')]

DS = os.environ.get('DS', 'ds01')
NPER = 200
HP = {'llke': dict(h=0.1), 'bspline': dict(n_knots=30),
      'lr': dict(), 'cabn': dict(lam1=0.0)}
MAKERS = {'llke': B2.LLKENM, 'bspline': B2.BSplineNM,
          'lr': B2.LinearNM, 'cabn': B2.CaBNNM}
RES = os.path.join(HERE, f'{DS}v6_test_results' + (f'_s{os.environ["SEEDS_ONLY"]}' if 'SEEDS_ONLY' in os.environ else '') + '.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load_gp(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def test_props(cache):
    At = cache['A_test']
    unit = At[:, 0].astype(int); cyc = At[:, 1].astype(int); hs = At[:, 3]
    props = {}
    for u in np.unique(unit):
        m = unit == u
        cyc_u = cyc[m]; hs_u = hs[m]
        ucyc = np.unique(cyc_u)
        hours = np.array([(cyc_u == c).sum() / 3600.0 for c in ucyc])
        hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
        below = np.where(hs_by < 0.5)[0]
        onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
        props[int(u)] = (ucyc, hours, onset)
    return props


def sample_grid_test(cache, sd, props):
    At = cache['A_test']
    unit = At[:, 0].astype(int); cyc = At[:, 1].astype(int)
    out = {}
    for u in sorted(props):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        idx, cc = [], []
        for c in props[u][0]:
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r); cc.append(np.full(len(r), c))
        out[u] = (np.concatenate(idx), np.concatenate(cc))
    return out


def build_test_stats(tag, sd, cache, props, grid):
    out_path = os.path.join(HERE, f'{DS}v6_{tag}_test_stats_s{sd}.npz')
    if os.path.exists(out_path):
        log(f'seed{sd} {tag}: test stats cached')
        return
    gp_path = os.path.join(HERE, f'{DS}v6_gp_{tag}_s{sd}.pt')
    gp = load_gp(gp_path)
    Wt, Xt = cache['W_test'], cache['X_s_test'][:, SIDX]
    t0 = time.time()
    out = {}
    for u in sorted(props):
        idx, cc = grid[u]
        pred, dcv, ll = point_stats_parallel(gp, Wt[idx], Xt[idx], gp_path=gp_path, log=log)
        ucyc, hours, onset = props[u]
        out[f'u{u}_cc'] = cc.astype(np.int32)
        out[f'u{u}_resid'] = (Xt[idx] - pred).astype(np.float32)
        out[f'u{u}_dc'] = dcv.astype(np.float32)
        out[f'u{u}_ll'] = ll.astype(np.float32)
        out[f'u{u}_ucyc'] = ucyc.astype(np.int32)
        out[f'u{u}_hours'] = hours.astype(np.float32)
        out[f'u{u}_onset'] = np.array([onset])
    np.savez(out_path, **out)
    log(f'seed{sd} {tag}: test stats written ({time.time()-t0:.0f}s)')


def test_tables(tag, sd):
    D = load_stats(DS, sd, f'{tag}_test')
    units = sorted(D)
    raw = {}
    for u in units:
        d = D[u]
        raw[u] = np.stack([trim25(d['resid'][d['cc'] == c])
                           for c in d['ucyc']])
    return units, raw


def per_level(P, T, F, nseeds):
    """per-level RMSE and NASA (sum per seed, mean over seeds)."""
    out = {}
    n = len(P) // nseeds
    for f in FRACS + ['all']:
        m = np.ones(len(P), bool) if f == 'all' else np.isclose(F, f)
        rm = float(np.sqrt(((P[m] - T[m]) ** 2).mean()))
        sc = np.mean([nasa(P[i * n:(i + 1) * n][m[i * n:(i + 1) * n]],
                           T[i * n:(i + 1) * n][m[i * n:(i + 1) * n]])
                      for i in range(nseeds)])
        out[f] = (rm, float(sc))
    return out


def fmt_levels(pl):
    return ' | '.join(f'{k if k == "all" else int(k*100)}%: '
                      f'{v[0]:.2f}/{v[1]:.1f}' for k, v in pl.items())

def build_bench_residuals(cache, props):
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    Wt, Xt = cache['W_test'], cache['X_s_test'][:, SIDX]
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    t_units = sorted(props)
    for sd in SEEDS:
        dev_p = os.path.join(HERE, f'{DS}v6_bench_dev_resid_s{sd}.npz')
        tst_p = os.path.join(HERE, f'{DS}v6_bench_test_resid_s{sd}.npz')
        if os.path.exists(dev_p) and os.path.exists(tst_p):
            log(f'  seed{sd}: bench residuals cached'); continue
        Zc = np.load(os.path.join(HERE, f'{DS}v6_c3_stats_s{sd}.npz'))
        tr = Zc['train_idx']
        Dd = load_stats(DS, sd, 'c3')
        d_idx, d_cc, d_uu = [], [], []
        for u in sorted(Dd):
            rows = np.where(unit == u)[0]; cyc_u = cyc[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            for c in Dd[u]['ucyc']:
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                d_idx.append(r); d_cc.append(np.full(len(r), c))
                d_uu.append(np.full(len(r), u))
        d_idx = np.concatenate(d_idx); d_cc = np.concatenate(d_cc)
        d_uu = np.concatenate(d_uu)
        grid = sample_grid_test(cache, sd, props)
        t_idx = np.concatenate([grid[u][0] for u in t_units])
        t_cc = np.concatenate([grid[u][1] for u in t_units])
        t_uu = np.concatenate([np.full(len(grid[u][0]), u) for u in t_units])
        dev_out = {'cc': d_cc, 'uu': d_uu}; tst_out = {'cc': t_cc, 'uu': t_uu}
        for name, maker in MAKERS.items():
            t0 = time.time()
            m = maker(**HP[name]); np.random.seed(0); m.fit(W[tr], X[tr])
            pred, _, _ = stats_parallel(m, W[d_idx], X[d_idx])
            dev_out[f'{name}_resid'] = (X[d_idx] - pred).astype(np.float32)
            pred_t, _, _ = stats_parallel(m, Wt[t_idx], Xt[t_idx])
            tst_out[f'{name}_resid'] = (Xt[t_idx] - pred_t).astype(np.float32)
            log(f'  seed{sd} {name}: residuals ready ({time.time()-t0:.0f}s)')
        np.savez(dev_p, **dev_out); np.savez(tst_p, **tst_out)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'{DS.upper()} V6 TEST — FIRST v6 test opening of this subset, '
        'pre-registered (onset gate check, HI#2/HI#1 RUL, benchmark)')
    Zr = np.load(os.path.join(HERE, f'{DS}v6_detsel.npz'))
    rule = dict(NB=int(Zr['NB'][0]), V_sparse=float(Zr['V_sparse'][0]),
                V_dense=float(Zr['V_dense'][0]), wval=float(Zr['wval'][0]),
                wmode=str(Zr['wmode'][0]), clip=float(Zr['clip'][0]))
    Zh1 = np.load(os.path.join(HERE, f'{DS}v6_hi1.npz'))
    Zh2 = np.load(os.path.join(HERE, f'{DS}v6_stage2.npz'))
    R1 = (float(Zh1['l1'][0]), float(Zh1['l2'][0]))
    R2 = (float(Zh2['l1'][0]), float(Zh2['l2'][0]))
    log(f'frozen: rule {rule}; HI#1 recipe {R1}; HI#2 recipe {R2}')

    # ---- dev side: retrain HI nets, sanity vs Phases 1c / 2 ----
    side = {}
    for tag, recipe, ref in (('c3', R1, Zh1['dev_rm']), ('hr', R2, Zh2['rms'])):
        cfg = dict(HI_CFG); cfg.update(lambda1=recipe[0], lambda2=recipe[1],
                                       end_target=1.03)
        for sd in SEEDS:
            units, raw, Zn, hrs, mu, sg = tables_ungated(DS, sd, tag)
            np.random.seed(sd)
            him, _ = train_model_tail([Zn[u] for u in units], **cfg)
            HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
            P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
            rm_sd = float(np.sqrt(((P - T) ** 2).mean()))
            side[(tag, sd)] = (him, mu, sg, HIs, units, b)
            log(f'SANITY {tag} seed{sd}: dev LOO {rm_sd:.2f} (frozen '
                f'{float(ref[sd]):.2f})')
            assert abs(rm_sd - float(ref[sd])) <= 0.02, \
                f'dev sanity FAILED ({tag} seed{sd})'

    # ---- test contact starts here ----
    cache = np.load(os.path.join(HERE, f'cache_{DS}.npz'))
    props = test_props(cache)
    t_units = sorted(props)
    log(f'test units {t_units}: ' + ', '.join(
        f'u{u}(L{len(props[u][0])}, {props[u][1].mean():.2f} h/cyc, '
        f'hs {props[u][2]})' for u in t_units))
    for sd in SEEDS:
        grid = sample_grid_test(cache, sd, props)
        for tag in ('c3', 'hr'):
            build_test_stats(tag, sd, cache, props, grid)
    if os.environ.get('STATS_ONLY'):
        build_bench_residuals(cache, props)
        log('STATS_ONLY — test stats + bench residuals built for '
            f'seeds {SEEDS}; analysis left to the merge run')
        return

    # (a) onset on test
    e_rule, e_ug, grp, ids = [], [], [], []
    for sd in SEEDS:
        D = load_stats(DS, sd, 'c3_test')
        for u in t_units:
            d = D[u]
            nb = nbase_at(d, rule['wval'], rule['wmode'])
            V = gate_V(d, rule)
            det_r = detect_p(q75_curve(d, V), d['dur'], d['ucyc'],
                             rule['wval'], rule['wmode'], rule['clip'])
            det_u = detect_p(q75_curve(d), d['dur'], d['ucyc'],
                             rule['wval'], rule['wmode'], rule['clip'])
            e_rule.append(det_r - d['onset']); e_ug.append(det_u - d['onset'])
            grp.append('sparse' if nb < rule['NB'] else 'dense')
            ids.append((sd, u))
    e_rule, e_ug = np.array(e_rule, float), np.array(e_ug, float)
    grp = np.array(grp)
    rm = lambda e: float(np.sqrt((e ** 2).mean())) if len(e) else np.nan
    try:
        p_on = f"{st.wilcoxon(np.abs(e_rule), np.abs(e_ug), zero_method='zsplit').pvalue:.3f}"
    except ValueError:
        p_on = '1.000 (identical)'
    log('')
    log(f'(a) ONSET test ({len(e_rule)} cases): conditional {rm(e_rule):.2f} '
        f'vs ungated {rm(e_ug):.2f}, paired p={p_on}')
    for g in ('sparse', 'dense'):
        m = grp == g
        if m.any():
            log(f'    {g} ({int(m.sum())} cases, units '
                f'{sorted({u for (sd, u), k in zip(ids, m) if k})}): '
                f'conditional {rm(e_rule[m]):.2f} vs ungated {rm(e_ug[m]):.2f}')
    log(f'    dev recap: conditional {float(Zr["r_rule"][0]):.2f} vs ungated '
        f'{float(Zr["r_ug"][0]):.2f}')
    for u in t_units:
        log(f'    u{u}: conditional errors '
            f'{[int(e) for (sd, uu), e in zip(ids, e_rule) if uu == u]}, '
            f'ungated {[int(e) for (sd, uu), e in zip(ids, e_ug) if uu == u]}')

    # (b) RUL: HI#2 and HI#1 chains on test
    results, preds = {}, {}
    for tag, name in (('hr', 'HI#2 (proposed)'), ('c3', 'HI#1 (cycle<=3)')):
        rms, nss, Ps, Ts, Fs = [], [], [], [], []
        for sd in SEEDS:
            him, mu, sg, HIs, units, b = side[(tag, sd)]
            tu, t_raw = test_tables(tag, sd)
            Zn_t = {u: (t_raw[u] - mu) / sg for u in tu}
            HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in tu}
            cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
            ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in tu}
            P, T, F, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
            rms.append(float(np.sqrt(((P - T) ** 2).mean())))
            nss.append(nasa(P, T)); Ps.append(P); Ts.append(T); Fs.append(F)
        P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
        pl = per_level(P, T, F, len(SEEDS))
        results[tag] = dict(rms=rms, nss=nss, pl=pl)
        preds[tag] = (P, T, F)
        log('')
        log(f'(b) {name}: test RUL {np.mean(rms):.2f} ± {np.std(rms):.2f}  '
            f'NASA {np.mean(nss):.1f} ± {np.std(nss):.1f}')
        log(f'    per level RMSE/score: {fmt_levels(pl)}')
    p_b = st.wilcoxon(np.abs(preds['hr'][0] - preds['hr'][1]),
                      np.abs(preds['c3'][0] - preds['c3'][1]),
                      zero_method='zsplit').pvalue
    log(f'    paired |err| HI#2 vs HI#1 ({len(preds["hr"][0])} preds, '
        f'unit n={len(t_units)}): p={p_b:.3f}')

    # (c) benchmark competitors on the stage-1 rows, HI#1 recipe downstream
    log('')
    log('(c) BENCHMARK — competitors on the identical cycle<=3 rows, '
        f'HI#1 recipe {R1} retrained per model, ungated')
    build_bench_residuals(cache, props)
    cfg1 = dict(HI_CFG); cfg1.update(lambda1=R1[0], lambda2=R1[1],
                                     end_target=1.03)
    def summaries(Z, name, hours_map):
        cc, uu, rs = Z['cc'], Z['uu'], Z[f'{name}_resid']
        raw, hrs = {}, {}
        for u in np.unique(uu):
            mm = uu == u; ucyc = np.unique(cc[mm])
            raw[int(u)] = np.stack([trim25(rs[mm & (cc == c)]) for c in ucyc])
            if hours_map is not None:
                hrs[int(u)] = np.cumsum(hours_map[int(u)][:len(ucyc)])
        return raw, hrs

    for name in MAKERS:
        rms, nss, drm, Ps, Ts, Fs, oks = [], [], [], [], [], [], []
        for sd in SEEDS:
            Zd = np.load(os.path.join(HERE, f'{DS}v6_bench_dev_resid_s{sd}.npz'))
            Zt = np.load(os.path.join(HERE, f'{DS}v6_bench_test_resid_s{sd}.npz'))
            Dd = load_stats(DS, sd, 'c3')
            d_raw, d_hrs = summaries(Zd, name, {u: Dd[u]['dur'] for u in Dd})
            t_raw, _ = summaries(Zt, name, None)
            du = sorted(d_raw); tu = sorted(t_raw)
            allr = np.concatenate([d_raw[u] for u in du])
            mu, sg = allr.mean(0), allr.std(0) + 1e-8
            Zn_d = {u: (d_raw[u] - mu) / sg for u in du}
            Zn_t = {u: (t_raw[u] - mu) / sg for u in tu}
            np.random.seed(sd)
            him, _ = train_model_tail([Zn_d[u] for u in du], **cfg1)
            HI_d = {u: med3(him.forward(Zn_d[u]).flatten()) for u in du}
            HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in tu}
            ok, _ = shape_ok(HI_d); oks.append(int(ok))
            Pd, Td, _, b, _ = loo('cycle', du, HI_d, d_hrs, BETA_C)
            drm.append(float(np.sqrt(((Pd - Td) ** 2).mean())))
            cd = {u: np.arange(len(HI_d[u])) / 500.0 for u in du}
            ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in tu}
            P, T, F, _ = eval_units(b, cd, HI_d, ct, HI_t, FRACS)
            rms.append(float(np.sqrt(((P - T) ** 2).mean())))
            nss.append(nasa(P, T)); Ps.append(P); Ts.append(T); Fs.append(F)
        P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
        pl = per_level(P, T, F, len(SEEDS))
        results[name] = dict(rms=rms, nss=nss, pl=pl, dev=drm, shape=sum(oks))
        preds[name] = (P, T, F)
        assert np.allclose(T, preds['hr'][1]), f'{name}: pairing off'
        p_c = st.wilcoxon(np.abs(P - T),
                          np.abs(preds['hr'][0] - preds['hr'][1]),
                          zero_method='zsplit').pvalue
        log(f'  {name}: dev {np.mean(drm):.2f} ± {np.std(drm):.2f} | test '
            f'{np.mean(rms):.2f} ± {np.std(rms):.2f}  NASA {np.mean(nss):.1f} '
            f'± {np.std(nss):.1f}  shape {sum(oks)}/3  vs HI#2 p={p_c:.3f}')
        log(f'    per level RMSE/score: {fmt_levels(pl)}')

    log('')
    log(f'SUMMARY {DS.upper()} (test units n={len(t_units)}, 3 seeds)')
    log(f'  onset: conditional {rm(e_rule):.2f} vs ungated {rm(e_ug):.2f} '
        f'(p={p_on}); dev {float(Zr["r_rule"][0]):.2f} vs '
        f'{float(Zr["r_ug"][0]):.2f}')
    for k, lab in [('hr', 'Proposed (HI#2)'), ('c3', 'MOGP (HI#1)'),
                   ('llke', 'LLKE'), ('bspline', 'B-spline'), ('lr', 'LR'),
                   ('cabn', 'CaBN')]:
        r = results[k]
        log(f'  {lab:18s} test {np.mean(r["rms"]):.2f} ± {np.std(r["rms"]):.2f}'
            f'  NASA {np.mean(r["nss"]):.1f} ± {np.std(r["nss"]):.1f}  '
            f'levels ' + ' / '.join(f'{r["pl"][f][0]:.2f}' for f in FRACS))
    np.savez(os.path.join(HERE, f'{DS}v6_test.npz'),
             e_on_rule=e_rule, e_on_ug=e_ug, on_grp=grp,
             on_ids=np.array(ids, int),
             **{f'{k}_{p}': v for k, (P_, T_, F_) in preds.items()
                for p, v in (('P', P_), ('T', T_), ('F', F_))},
             **{f'{k}_t_rm': np.array(results[k]['rms']) for k in results},
             **{f'{k}_t_ns': np.array(results[k]['nss']) for k in results})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
