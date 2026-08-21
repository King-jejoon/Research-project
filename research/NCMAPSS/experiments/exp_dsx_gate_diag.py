"""exp_dsx_gate_diag.py — WHY does the confidence gate help on DS03 but not on
DS08a / DS07?  Dev-only re-scoring of cached stats (no test contact).
  (1) WIDE V sweep: absolute detcov axis from below min (= ungated) to the
      maximum, 120 steps, each dataset at its own frozen window/clip;
      overall + sparse/dense group curves; best-vs-ungated.
  (2) Mechanism: per unit, baseline sigma0 gated/ungated ratio at the
      frozen rule, baseline cycle count, fraction of points kept, and the
      likelihood-drop SNR (baseline mean - post-onset mean)/sigma0.
Output: dsx_gate_diag_results.txt"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from exp_dsx_lib import load_stats, q75_curve, detect_p, nbase_at
from exp_v4_setcmp import load as load_ds03
SEEDS = [int(x) for x in os.environ.get('SEEDS_ONLY', '0,1,2').split(',')]
DSLIST = os.environ.get('DSLIST', 'ds03,ds08a,ds07').split(',')
RES = os.path.join(HERE, os.environ.get('DIAG_OUT', 'dsx_gate_diag_results.txt')); open(RES, 'w').close()
def log(s):
    print(s, flush=True); open(RES, 'a').write(s + '\n')

def ds03_stats(sd):
    D = load_ds03(5, sd)
    return {int(u): dict(cc=d['cc'], dc=d['dc'], ll=d['ll'], ucyc=np.asarray(d['ucyc'], int),
                         dur=np.asarray(d['dur'], float), onset=int(d['onset'])) for u, d in D.items()}

def cfg_for(ds):
    if ds == 'ds03':
        return dict(load=ds03_stats, wval=30.0, wmode='fh', clip=10.0, NB=10, Vs=19.0, Vd=20.2)
    Z = np.load(os.path.join(HERE, f'{ds}v6_detsel.npz'))
    return dict(load=lambda sd, ds=ds: load_stats(ds, sd), wval=float(Z['wval'][0]),
                wmode=str(Z['wmode'][0]), clip=float(Z['clip'][0]), NB=int(Z['NB'][0]),
                Vs=float(Z['V_sparse'][0]), Vd=float(Z['V_dense'][0]))
CFG = {ds: cfg_for(ds) for ds in DSLIST}

def baseline_w(d, wval, wmode):
    return nbase_at(d, wval, wmode)

for name, c in CFG.items():
    D = {sd: c['load'](sd) for sd in SEEDS}
    units = sorted(D[0]); w, m, cl = c['wval'], c['wmode'], c['clip']
    nb = {u: baseline_w(D[0][u], w, m) for u in units}
    sparse = [u for u in units if nb[u] < c['NB']]; dense = [u for u in units if nb[u] >= c['NB']]
    all_dc = np.concatenate([D[sd][u]['dc'] for sd in SEEDS for u in units])
    lo, hi = all_dc.min(), all_dc.max()
    VGRID = [None] + list(np.linspace(lo - 1e-3, hi, 120))
    log(f'\n================ {name.upper()}  window {w}{m} clip {cl}  '
        f'detcov range [{lo:.2f}, {hi:.2f}]  nbase {nb}  sparse{sparse} dense{dense}')

    def errs(V, subset):
        e = []
        for sd in SEEDS:
            for u in subset:
                d = D[sd][u]
                e.append(detect_p(q75_curve(d, V), d['dur'], d['ucyc'], w, m, cl) - d['onset'])
        return np.array(e, float)
    rm = lambda e: float(np.sqrt((e ** 2).mean())) if len(e) else np.nan
    curve = {}
    for V in VGRID:
        curve[V] = (rm(errs(V, units)), rm(errs(V, sparse)) if sparse else np.nan,
                    rm(errs(V, dense)) if dense else np.nan)
    ug = curve[None]
    log(f'(1) WIDE V SWEEP — ungated: all {ug[0]:.2f} sparse {ug[1]:.2f} dense {ug[2]:.2f}')
    Vs_ = [V for V in VGRID if V is not None]
    for j, lab in enumerate(['all', 'sparse', 'dense']):
        vals = np.array([curve[V][j] for V in Vs_])
        if np.all(np.isnan(vals)): continue
        k = int(np.nanargmin(vals)); V = Vs_[k]
        frac = float(np.mean(all_dc >= V))
        better = [Vv for Vv, vv in zip(Vs_, vals) if vv < curve[None][j] - 1e-9]
        log(f'    {lab:6s}: best V={V:6.2f} (keeps {frac*100:4.1f}% of points) RMSE={vals[k]:.2f} '
            f'vs ungated {curve[None][j]:.2f}; V values beating ungated: {len(better)}/{len(Vs_)}'
            + (f' (range {min(better):.2f}..{max(better):.2f})' if better else ''))
    # coarse print of the overall curve at percentiles of the axis
    qs = [0, 10, 25, 50, 75, 90, 95, 98, 99, 99.5]
    log('    overall RMSE along the axis (percentile of pooled detcov -> V -> RMSE all/sparse/dense):')
    for q in qs:
        V = float(np.percentile(all_dc, q)); k = int(np.argmin(np.abs(np.array(Vs_) - V)))
        r = curve[Vs_[k]]
        log(f'      P{q:>4}: V={Vs_[k]:6.2f}  {r[0]:6.2f} / {r[1]:6.2f} / {r[2]:6.2f}')

    # (2) mechanism at the frozen rule
    log('(2) MECHANISM at the frozen rule (per unit, mean over seeds):')
    log('      unit  nbase  group   kept%   sigma0 gated/ungated   drop-SNR ungated -> gated   onset err ungated -> gated')
    for u in units:
        V = c['Vs'] if nb[u] < c['NB'] else c['Vd']
        kept, s_ratio, snr_u, snr_g, e_u, e_g = [], [], [], [], [], []
        for sd in SEEDS:
            d = D[sd][u]
            cu, cg = q75_curve(d), q75_curve(d, V)
            wb = baseline_w(d, w, m)
            s0u, s0g = cu[:wb].std(ddof=1) + 1e-8, cg[:wb].std(ddof=1) + 1e-8
            on_i = int(np.searchsorted(d['ucyc'], d['onset']))
            post_u = cu[on_i:on_i + 10].mean() if on_i < len(cu) else np.nan
            post_g = cg[on_i:on_i + 10].mean() if on_i < len(cg) else np.nan
            kept.append(float((d['dc'] >= V).mean()))
            s_ratio.append(s0g / s0u)
            snr_u.append((cu[:wb].mean() - post_u) / s0u); snr_g.append((cg[:wb].mean() - post_g) / s0g)
            e_u.append(detect_p(cu, d['dur'], d['ucyc'], w, m, cl) - d['onset'])
            e_g.append(detect_p(cg, d['dur'], d['ucyc'], w, m, cl) - d['onset'])
        log(f'      u{u:<3d} {nb[u]:5d}  {"sparse" if u in sparse else "dense ":6s} {np.mean(kept)*100:5.1f}   '
            f'{np.mean(s_ratio):6.2f}x              {np.mean(snr_u):6.1f} -> {np.mean(snr_g):6.1f}          '
            f'{np.mean(e_u):+6.1f} -> {np.mean(e_g):+6.1f}')
log('\ndone')
