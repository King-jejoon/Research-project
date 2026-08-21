"""
exp_dsx_detsel.py — DS01/DS02 transfer, PHASE 1b: detector re-selection.
Dev only, cached stats, no test contact.  Every numeric detector constant
is re-selected here with the DS03 procedures; only the structure is fixed
(gated q75-LL curve -> baseline window -> clip mu0 - c*sd0 -> mean-drop;
coverage-conditional gate triggered by the observable baseline cycle count).

Order (DS03-faithful):
  (A) baseline window x clip 2-D sweep on UNGATED curves — grids are the
      DS03 grids extended past their minima (sweep-to-collapse rule):
      windows {3,4,5,6,8,12,20}cy + {10,15,20,25,30,50,80}h,
      clips {4,6,8,10,12,14,16,20} sigma; select min RMSE, one-seed-sd
      tie band disclosed.
  (B) gate V absolute sweep at the selected (window, clip): axis =
      percentiles of the pooled dev detcov (absolute values, swept past
      the minimum to collapse at both ends); scored overall and stratified
      by baseline cycle count nbase (the deployable trigger).
      CONDITIONAL-GATE STRUCTURE IS MANDATORY (user decision 2026-08-19):
      the deployed rule keeps the DS03 form nbase < NB -> V_sparse else
      V_dense.  Because the nbase trigger is only observable under a
      flight-hour baseline, the deployed window is the best FLIGHT-HOUR
      window inside the (A) tie band (best fh overall if none lands in
      the band — disclosed); the raw (A) minimum stays logged.  NB comes
      from the largest nbase gap (>= 3 cycles, both sides >= 2 units);
      if dev holds a single class, NB carries the DS03 structural value
      10 and both branches share the dev-derived V (uncalibratable
      branch disclosed).  Paired Wilcoxon vs ungated over 18 dev cases.
  (C) confirmation re-sweep of window x clip on the CONDITIONAL curves
      (v5_sens analogue) — tie-band check only, selection stays from (A)
      unless it leaves the band (disclosed).
Usage: DS=ds01|ds02 python3 exp_dsx_detsel.py
Outputs: {DS}v6_detsel_results.txt, {DS}v6_detsel.npz (frozen rule)
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_dsx_lib import load_stats, q75_curve, detect_p, nbase_at
SEEDS = [int(x) for x in os.environ.get('SEEDS_ONLY', '0,1,2').split(',')]

DS = os.environ.get('DS', 'ds01')
assert DS in ('ds01', 'ds02', 'ds07', 'ds08a', 'ds04', 'ds05', 'ds06', 'ds08c')
WINDOWS = [('cyc', w) for w in (3, 4, 5, 6, 8, 12, 20)] + \
          [('fh', w) for w in (10, 15, 20, 25, 30, 50, 80)]
CLIPS = [4, 6, 8, 10, 12, 14, 16, 20]
RES = os.path.join(HERE, f'{DS}v6_detsel_results.txt')
NPZ_SUFFIX = ''


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def sweep_windows(D, curves, label):
    """window x clip sweep on the given per-(sd,u) curves.
    returns dict[(wmode,wval,clip)] = (pooled rmse, per-seed rmse)."""
    out = {}
    for wmode, wval in WINDOWS:
        for clip in CLIPS:
            per_seed = []
            for sd in SEEDS:
                e = []
                for u, d in D[sd].items():
                    det = detect_p(curves[(sd, u)], d['dur'], d['ucyc'],
                                   wval, wmode, clip)
                    e.append(det - d['onset'])
                per_seed.append(float(np.sqrt(np.mean(np.square(e)))))
            allc = np.concatenate([[detect_p(curves[(sd, u)], d['dur'],
                                             d['ucyc'], wval, wmode, clip)
                                    - d['onset']
                                    for u, d in D[sd].items()]
                                   for sd in SEEDS])
            out[(wmode, wval, clip)] = (float(np.sqrt((allc ** 2).mean())),
                                        per_seed)
    best = min(out, key=lambda k: out[k][0])
    band_w = float(np.std(out[best][1], ddof=1)) if len(SEEDS) > 1 else 0.0
    band = sorted([k for k in out if out[k][0] <= out[best][0] + band_w],
                  key=lambda k: out[k][0])
    log(f'{label}: minimum {best} RMSE={out[best][0]:.2f} '
        f'(seed sd {band_w:.2f})')
    log(f'{label}: tie band ({len(band)}): ' +
        ', '.join(f'{k}={out[k][0]:.2f}' for k in band[:12]) +
        (' ...' if len(band) > 12 else ''))
    return out, best, band


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'{DS.upper()} V6 DETSEL — detector re-selection, DS03 procedures, '
        'dev only')
    D = {sd: load_stats(DS, sd) for sd in SEEDS}
    units = sorted(D[SEEDS[0]])
    for u in units:
        d = D[0][u]
        log(f'  u{u}: mean {d["dur"].mean():.2f} h/cycle, hs-onset '
            f'{d["onset"]}, len {len(d["ucyc"])}')

    # (A) ungated window x clip
    ug = {(sd, u): q75_curve(D[sd][u]) for sd in SEEDS for u in D[sd]}
    log('')
    log('(A) window x clip sweep, UNGATED curves')
    outA, bestA, bandA = sweep_windows(D, ug, 'A')

    # deployed window: best flight-hour window (conditional trigger needs
    # an hour baseline) inside the tie band, else best fh overall
    fh_band = [k for k in bandA if k[0] == 'fh']
    if fh_band:
        wmode, wval, clip = fh_band[0]
        log(f'  deployed window = best fh in tie band: '
            f'{(wmode, wval, clip)} RMSE={outA[(wmode, wval, clip)][0]:.2f}')
    else:
        fh_all = {k: v for k, v in outA.items() if k[0] == 'fh'}
        wmode, wval, clip = min(fh_all, key=lambda k: fh_all[k][0])
        log(f'  WARNING: no fh window in the tie band — deployed window = '
            f'best fh overall {(wmode, wval, clip)} '
            f'RMSE={outA[(wmode, wval, clip)][0]:.2f} '
            f'(raw minimum {bestA} stays logged)')

    # (B) V absolute sweep at the deployed window/clip
    all_dc = np.concatenate([D[sd][u]['dc'] for sd in SEEDS for u in D[sd]])
    VGRID = np.unique(np.round(np.percentile(
        all_dc, np.linspace(2, 99.8, 40)), 2))
    nb = {u: nbase_at(D[0][u], wval, wmode) for u in units}
    log('')
    log(f'(B) gate V absolute sweep at {(wmode, wval, clip)}; '
        f'per-unit nbase {nb}')
    nbv = sorted(set(nb.values()))
    NB = None
    if len(nbv) > 1:
        gaps = np.diff(nbv)
        gi = int(np.argmax(gaps))
        lo_t = [u for u in units if nb[u] <= nbv[gi]]
        hi_t = [u for u in units if nb[u] > nbv[gi]]
        if gaps[gi] >= 3 and len(lo_t) >= 2 and len(hi_t) >= 2:
            NB = int(round((nbv[gi] + nbv[gi + 1]) / 2))
            lo_g, hi_g = lo_t, hi_t
            log(f'  nbase gap {gaps[gi]} -> groups sparse{lo_g} '
                f'dense{hi_g}, NB={NB}')
    if NB is None:
        NB = 10          # DS03 structural threshold, uncalibratable here
        lo_g = [u for u in units if nb[u] < NB]
        hi_g = [u for u in units if nb[u] >= NB]
        log(f'  single dev class — NB carries the DS03 structural value '
            f'{NB}; groups sparse{lo_g} dense{hi_g} (one branch '
            'uncalibratable, shares the dev-derived V)')

    def v_errors(V, subset):
        e = []
        for sd in SEEDS:
            for u in subset:
                d = D[sd][u]
                cur = q75_curve(d, V)
                e.append(detect_p(cur, d['dur'], d['ucyc'], wval, wmode,
                                  clip) - d['onset'])
        return np.array(e, float)

    rows = {}
    for V in VGRID:
        r_all = float(np.sqrt((v_errors(V, units) ** 2).mean()))
        r_lo = float(np.sqrt((v_errors(V, lo_g) ** 2).mean())) if lo_g else \
            np.nan
        r_hi = float(np.sqrt((v_errors(V, hi_g) ** 2).mean())) if hi_g else \
            np.nan
        rows[V] = (r_all, r_lo, r_hi)
    e_ug = np.concatenate([[detect_p(ug[(sd, u)], D[sd][u]['dur'],
                                     D[sd][u]['ucyc'], wval, wmode, clip)
                            - D[sd][u]['onset'] for u in units]
                           for sd in SEEDS]).astype(float)
    r_ug = float(np.sqrt((e_ug ** 2).mean()))
    log(f'  ungated reference at {(wmode, wval, clip)}: RMSE={r_ug:.2f}')
    for V in VGRID:
        r = rows[V]
        log(f'  V={V:7.2f}: all={r[0]:6.2f}  sparse={r[1]:6.2f}  '
            f'dense={r[2]:6.2f}')
    if hi_g:
        Vd = min(rows, key=lambda V: rows[V][2])
        Vs = min(rows, key=lambda V: rows[V][1]) if lo_g else Vd
    else:
        Vs = min(rows, key=lambda V: rows[V][1])
        Vd = Vs
    log(f'  dense-branch V={Vd:.2f}' +
        (f' (RMSE {rows[Vd][2]:.2f})' if hi_g else ' (uncalibratable, '
         'shares sparse V)') +
        f'; sparse-branch V={Vs:.2f}' +
        (f' (RMSE {rows[Vs][1]:.2f})' if lo_g else ' (uncalibratable, '
         'shares dense V)'))
    rule = dict(NB=NB, V_sparse=float(Vs), V_dense=float(Vd),
                wval=float(wval), wmode=wmode, clip=float(clip))

    def rule_errors(subset):
        e = []
        for sd in SEEDS:
            for u in subset:
                d = D[sd][u]
                V = Vs if (NB >= 0 and nb[u] < NB) else Vd
                cur = q75_curve(d, V)
                e.append(detect_p(cur, d['dur'], d['ucyc'], wval, wmode,
                                  clip) - d['onset'])
        return np.array(e, float)

    e_rule = rule_errors(units)
    r_rule = float(np.sqrt((e_rule ** 2).mean()))
    try:
        p = st.wilcoxon(np.abs(e_rule), np.abs(e_ug),
                        zero_method='zsplit').pvalue
        p = f'{p:.3f}'
    except ValueError:
        p = '1.000 (identical)'
    log(f'  FROZEN RULE: {rule}')
    log(f'  dev onset RMSE: rule={r_rule:.2f} vs ungated={r_ug:.2f} '
        f'(paired p={p}, {len(e_rule)} cases)')
    for u in units:
        eu = [rule_errors([u])[i] for i in range(len(SEEDS))]
        log(f'    u{u}: rule errors {[int(x) for x in eu]}')

    # (C) confirmation re-sweep on conditional curves
    cond = {}
    for sd in SEEDS:
        for u in units:
            d = D[sd][u]
            V = Vs if (NB >= 0 and nb[u] < NB) else Vd
            cond[(sd, u)] = q75_curve(d, V)
    log('')
    log('(C) confirmation window x clip sweep on CONDITIONAL curves')
    outC, bestC, bandC = sweep_windows(D, cond, 'C')
    inband = any(k == (wmode, wval, clip) for k in bandC)
    log(f'  deployed window {(wmode, wval, clip)} '
        f'{"stays inside" if inband else "LEAVES"} the '
        f'conditional-curve tie band (min {bestC})')

    np.savez(os.path.join(HERE, f'{DS}v6_detsel.npz'),
             wmode=np.array([wmode]), wval=np.array([wval]),
             clip=np.array([clip]), NB=np.array([NB]),
             V_sparse=np.array([rule['V_sparse']]),
             V_dense=np.array([rule['V_dense']]),
             nbase=np.array([[u, nb[u]] for u in units]),
             r_rule=np.array([r_rule]), r_ug=np.array([r_ug]))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
