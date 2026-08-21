"""
exp_v4_c3_detcov.py — v4 pilot: widen the training window to cycle <= 3 and
re-test the detcov gate's validity BEFORE running the full chain.

Coverage-dependence context (frozen results):
  cycle<3 (v3 chain) : gate significant, onset 8.68 -> 7.16 at V*=20.0, p=0.002
  cycle<5 (old chain): gate invalid, p>0.3
This script probes the intermediate point cycle<=3 (cycles 1,2,3).

Configuration (everything except the window frozen from v3):
  sensors  : T30, T48, T50, Nc, Wf (5)   kernel: RBF, coregionalization rank 1
  training : 1000 rows per (unit, cycle), cycles 1..3, 9 dev units = 27,000
  GP       : Vecchia m=18, Adam lr 0.1 x 120, mcond=15, 3 seeds
  detector : q75 cycle summary, 30 flight-hour baseline, clip mu0-10*sigma0,
             mean-drop (identical to v3, no re-tuning)

Validity test: onset RMSE over 9 dev units x 3 seeds, gate off vs an
ABSOLUTE-value detcov sweep (0.1 grid spanning the pooled p2..p60 region;
percentiles are used only to place the grid and to report kept%, never as the
axis).  Every V paired against no-gate with Wilcoxon on |error|; the tie band
(RMSE within 0.1 of the minimum) is printed instead of auto-selecting.

Outputs: v4_c3_train_s{sd}.npz, v4_c3_stats_s{sd}.npz, v4_c3_results.txt,
         fig_v4_c3_gate_sweep.png (paper dir)
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
import gpytorch
from sklearn.preprocessing import StandardScaler
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2
from exp_resid_clean import meandrop

DT = torch.float32
SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
TRAIN_CYC_MAX = 3             # cycle <= 3  (v3 was cycle < 3)
NPER_TRAIN, NPER = 1000, 200
M, STEPS, LR, MCOND = 18, 120, 0.1, 15
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'v4_c3_results.txt')
OUT = __import__('exp_paths').PAPER


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit(Wtr, Ytr, tries=4):
    for t in range(tries):
        xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
        tX = torch.tensor(xs.transform(Wtr), dtype=DT)
        tY = torch.tensor(ys.transform(Ytr), dtype=DT)
        model = GPyTorchMOGP(4, num_tasks=len(SENS), rank=1,
                             kernel='rbf').to('cpu', DT)
        lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=len(SENS)).to('cpu', DT)
        try:
            r = train_vecchia_mogp(model, lik, tX, tY, m=M, num_steps=STEPS,
                                   lr=LR, group=True, verbose=False)
            model.eval(); lik.eval()
            return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure,
                        xs=xs, ys=ys), t
        except torch._C._LinAlgError:
            rg = np.random.default_rng(1234 + t)
            keep = rg.permutation(len(Wtr))[:len(Wtr) - 50]
            Wtr, Ytr = Wtr[keep], Ytr[keep]
    raise RuntimeError('fit failed after retries')


@torch.no_grad()
def point_stats(gp, Wq, Xq, chunk=4096):
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                              gp['tY'], gp['struct'], teX,
                                              m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def build_stats():
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int); hs = A[:, 3]
    pool = np.where(cyc <= TRAIN_CYC_MAX)[0]

    for sd in SEEDS:
        out_path = os.path.join(HERE, f'v4_c3_stats_s{sd}.npz')
        if os.path.exists(out_path):
            log(f'seed{sd}: stats cached, skipping'); continue
        rng = np.random.default_rng(sd)
        tr = []
        for u in np.unique(unit[pool]):
            for c in range(1, TRAIN_CYC_MAX + 1):
                rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
                tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)),
                                     replace=False))
        tr = np.concatenate(tr)
        np.savez(os.path.join(HERE, f'v4_c3_train_s{sd}.npz'), train_idx=tr)
        log(f'seed{sd}: train={len(tr)} rows '
            f'({len(np.unique(unit[tr]))} units x {TRAIN_CYC_MAX} cycles '
            f'x {NPER_TRAIN})')
        t0 = time.time()
        gp, nret = fit(W[tr], X[tr])
        log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
        out = {}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]; hs_u = hs[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ucyc = np.unique(cyc_u)
            idx, cc = [], []
            for c in ucyc:
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                idx.append(r); cc.append(np.full(len(r), c))
            idx = np.concatenate(idx); cc = np.concatenate(cc)
            pred, dcv, ll = point_stats(gp, W[idx], X[idx])
            hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
            below = np.where(hs_by < 0.5)[0]
            onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
            out[f'u{u}_cc'] = cc.astype(np.int32)
            out[f'u{u}_resid'] = (X[idx] - pred).astype(np.float32)
            out[f'u{u}_dc'] = dcv.astype(np.float32)
            out[f'u{u}_ll'] = ll.astype(np.float32)
            out[f'u{u}_onset'] = np.array([onset])
        np.savez(out_path, **out)
        log(f'seed{sd}: stats cached ({time.time()-t0:.0f}s total)')


def load(sd):
    Z = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    out = {}
    for u in units:
        cc = Z[f'u{u}_cc']; ll = Z[f'u{u}_ll']; dc = Z[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        out[u] = dict(cc=cc, ll=ll, dc=dc, ucyc=ucyc,
                      dur=np.array([durs[pos[int(c)]] for c in ucyc], float),
                      onset=int(Z[f'u{u}_onset'][0]))
    return out


def detect(cur, dur, ucyc):
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def errors(D, V):
    e = []
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b if V is None else (b & (d['dc'] >= V))
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


def analyse():
    D = {sd: load(sd) for sd in SEEDS}
    pooled = np.concatenate([D[sd][u]['dc'] for sd in SEEDS for u in D[sd]])
    log('')
    log('pooled dev detcov distribution (descriptive only):')
    qs = [1, 2, 5, 10, 25, 50, 75, 90]
    log('  ' + '  '.join(f'p{q}={np.percentile(pooled, q):.2f}' for q in qs))

    e_off = errors(D, None)
    r_off = np.sqrt((e_off ** 2).mean())
    log('')
    log(f'no gate: onset RMSE={r_off:.2f}  delay={e_off.mean():+.2f}  (n=27)')
    log(f'  [v3 cycle<3 reference: no-gate 8.68 -> V*=20.0 gives 7.16, p=0.002;'
        f' cycle<5 chain: gate invalid p>0.3]')

    lo = np.floor(np.percentile(pooled, 2) * 10) / 10
    hi = np.ceil(np.percentile(pooled, 60) * 10) / 10
    grid = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)
    log('')
    log(f'absolute-axis sweep V={grid[0]}..{grid[-1]} step 0.1 '
        f'({len(grid)} values)')
    log(f'  {"V":>6} | {"kept%":>6} | {"RMSE":>6} | {"delay":>6} | '
        f'{"better/worse":>12} | p(Wilcoxon)')
    rows = []
    for V in grid:
        e = errors(D, float(V))
        rm = np.sqrt((e ** 2).mean())
        pw = st.wilcoxon(np.abs(e), np.abs(e_off), zero_method='zsplit').pvalue
        b = int((np.abs(e) < np.abs(e_off)).sum())
        w = int((np.abs(e) > np.abs(e_off)).sum())
        kept = 100.0 * (pooled >= V).mean()
        rows.append((float(V), rm, pw, kept, e.mean()))
        log(f'  {V:6.1f} | {kept:5.1f}% | {rm:6.2f} | {e.mean():+6.2f} | '
            f'{b:>5} /{w:>5} | {pw:.4f}')

    rmin = min(r[1] for r in rows)
    band = [r for r in rows if r[1] <= rmin + 0.1]
    nsig = [r for r in rows if r[2] < 0.05 and r[1] < r_off]
    log('')
    log(f'tie band (RMSE <= min+0.1): '
        + ', '.join(f'V={r[0]:.1f} (RMSE {r[1]:.2f}, p={r[2]:.3f})'
                    for r in band))
    if nsig:
        log(f'significant improvements (p<0.05 & RMSE<no-gate): '
            + ', '.join(f'V={r[0]:.1f}' for r in nsig))
        log(f'VERDICT: gate VALID at cycle<=3 '
            f'(best {rmin:.2f} vs no-gate {r_off:.2f})')
    else:
        log('significant improvements: NONE')
        log(f'VERDICT: gate NOT validated at cycle<=3 '
            f'(best {rmin:.2f} vs no-gate {r_off:.2f}) — '
            f'consistent with coverage-dependence if RMSE gain is absent')

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.axhline(r_off, color='gray', lw=1.2, label=f'no gate ({r_off:.2f})')
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker='o', ms=3.5,
            ls='--', color='tab:blue', label='gated')
    for r in rows:
        if r[2] < 0.05:
            ax.plot(r[0], r[1], marker='*', ms=13, color='tab:red')
    ax.set_xlabel('absolute detcov threshold V (keep detcov >= V)')
    ax.set_ylabel('Onset RMSE [cycles]')
    ax.set_title('Gate sweep, training window cycle<=3 (* p<0.05 vs no gate)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(frameon=False)
    fig.tight_layout()
    p_fig = os.path.join(OUT, 'fig_v4_c3_gate_sweep.png')
    fig.savefig(p_fig, dpi=150); plt.close(fig)
    log(f'figure: {p_fig}')


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('V4 PILOT — cycle<=3 window (27k), detcov gate validity test')
    build_stats()
    analyse()
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
