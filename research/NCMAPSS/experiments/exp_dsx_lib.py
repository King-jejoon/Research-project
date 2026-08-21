"""
exp_dsx_lib.py — shared loaders for the DS01/DS02 transfer chain (v6).
Dev-side only; test loaders live in the pre-registered test script.
DS is chosen per call (ds01|ds02), stats files are {DS}v6_c3_stats_s{sd}.npz.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_hi import trim25          # chain trim25, byte-identical

SEEDS = [0, 1, 2]


def load_stats(ds, sd, tag='c3'):
    Z = np.load(os.path.join(HERE, f'{ds}v6_{tag}_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    out = {}
    for u in units:
        out[u] = dict(cc=Z[f'u{u}_cc'], resid=Z[f'u{u}_resid'],
                      dc=Z[f'u{u}_dc'], ll=Z[f'u{u}_ll'],
                      ucyc=Z[f'u{u}_ucyc'].astype(int),
                      dur=Z[f'u{u}_hours'].astype(float),
                      onset=int(Z[f'u{u}_onset'][0]))
    return out


def q75_curve(d, V=None):
    """per-cycle q75 of ll; V=None -> ungated, else detcov gate at V."""
    cur = np.empty(len(d['ucyc']))
    for i, c in enumerate(d['ucyc']):
        b = d['cc'] == c
        m = b if V is None else (b & (d['dc'] >= V))
        if not m.any():
            m = b
        cur[i] = np.percentile(d['ll'][m], 75)
    return cur


def meandrop(x, kmin=3):
    n = len(x); s = np.std(x, ddof=1) + 1e-12; cs = np.cumsum(x)
    best = (None, -np.inf)
    for k in range(kmin, n - 4):
        m1 = cs[k - 1] / k; m2 = (cs[-1] - cs[k - 1]) / (n - k)
        t = np.sqrt(k * (n - k) / n) * (m1 - m2) / s
        if t > best[1]:
            best = (k, t)
    return best[0]


def detect_p(cur, dur, ucyc, wval, wmode, clip):
    """baseline window in cycles (wmode='cyc') or flight-hours ('fh'),
    clip = c*sigma; conventions identical to exp_detector_sens.detect."""
    if wmode == 'cyc':
        w = max(5, min(int(wval), len(cur) - 5))
    else:
        w = max(5, int(np.searchsorted(np.cumsum(dur), float(wval)) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - clip * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else \
        int(ucyc[-1] + 1)


def nbase_at(d, wval, wmode):
    """baseline cycle count under the window (the deployable trigger)."""
    if wmode == 'cyc':
        return max(5, min(int(wval), len(d['ucyc']) - 5))
    return max(5, int(np.searchsorted(np.cumsum(d['dur']), float(wval)) + 1))


def gate_V(d, rule):
    """rule = dict(NB, V_sparse, V_dense, wval, wmode); NB<0 -> single V."""
    if rule['NB'] < 0:
        return rule['V_dense']
    nb = nbase_at(d, rule['wval'], rule['wmode'])
    return rule['V_sparse'] if nb < rule['NB'] else rule['V_dense']


def tables_ungated(ds, sd, tag='c3'):
    """per-flight trim25 tables + pooled z-norm + cumulative hours."""
    D = load_stats(ds, sd, tag)
    units = sorted(D)
    raw, hrs = {}, {}
    for u in units:
        d = D[u]
        T = np.empty((len(d['ucyc']), d['resid'].shape[1]))
        for i, c in enumerate(d['ucyc']):
            T[i] = trim25(d['resid'][d['cc'] == c])
        raw[u] = T
        hrs[u] = np.cumsum(d['dur'])
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, raw, {u: (raw[u] - mu) / sg for u in units}, hrs, mu, sg


def run_grid(data, label, log, L1S=(2.0, 4.0, 6.0, 8.0, 12.0, 16.0),
             L2S=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """lambda1 x lambda2 grid, protocol = exp_v5c_lambda_grid (med3-free;
    shape 3/3 eligibility -> composite of min-max normalized Mon/Curv/Rng).
    data[sd] = (units, Zn, hrs).  Returns R, selected combo, comp dict."""
    from exp_rul_r23 import HI_CFG, BETA_C, loo, train_model_tail, shape_ok
    from exp_v4_hi import properties
    R = {}
    for l1 in L1S:
        for l2 in L2S:
            cfg = dict(HI_CFG); cfg.update(lambda1=l1, lambda2=l2,
                                           end_target=1.03)
            props, rmses, oks = [], [], []
            for sd in sorted(data):
                units, Zn, hrs = data[sd]
                np.random.seed(sd)
                him, _ = train_model_tail([Zn[u] for u in units], **cfg)
                HIs = {u: him.forward(Zn[u]).flatten() for u in units}
                ok, _ = shape_ok(HIs)
                P, T, F, bsel, _ = loo('cycle', units, HIs, hrs, BETA_C)
                props.append(properties(HIs))
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                oks.append(ok)
            pr = np.array(props).mean(0)
            R[(l1, l2)] = dict(mon=pr[0], curv=pr[1], rng=pr[2],
                               shift=pr[3], rul=float(np.mean(rmses)),
                               rul_sd=float(np.std(rmses)),
                               shape=int(np.sum(oks)))
            log(f'  l1={l1:>4} l2={l2:>4}: Mon={pr[0]:8.2f} '
                f'Curv={pr[1]:6.2f} Range={pr[2]:5.3f} Shift={pr[3]:6.4f} '
                f'LOO RUL={np.mean(rmses):.2f}±{np.std(rmses):.2f} '
                f'shape {int(np.sum(oks))}/{len(data)}')
    ok_keys = [k for k in R if R[k]['shape'] == len(data)]
    pool_keys = ok_keys if ok_keys else list(R)
    arr = {p: np.array([R[k][p] for k in pool_keys])
           for p in ['mon', 'curv', 'rng']}

    def norm(v):
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    comp = (norm(arr['mon']) + norm(arr['curv']) + norm(arr['rng'])) / 3
    order = np.argsort(-comp)
    log('')
    log(f'{label}: shape-passing combos ({len(ok_keys)}): {sorted(ok_keys)}')
    log(f'{label}: composite ranking top 6:')
    for i in order[:6]:
        k = pool_keys[i]
        log(f'  {k}: comp={comp[i]:.3f}  LOO RUL={R[k]["rul"]:.2f}')
    sel = pool_keys[int(order[0])]
    comp_by_key = {pool_keys[i]: float(comp[i])
                   for i in range(len(pool_keys))}
    return R, sel, comp_by_key


def point_stats_robust(gp, Wq, Xq, chunk=4096, log=None):
    """exp_v4_c3_detcov.point_stats with jitter escalation ON FAILURE ONLY:
    each chunk is tried at the chain default (1e-6); a Cholesky failure
    escalates to 1e-5 -> 1e-4 -> 1e-3 for that chunk and is logged.
    Chunks that succeed at the default are byte-identical to the chain."""
    import torch
    from exp_v4_c3_detcov import DT, MCOND
    from demo_cond2 import conditional_stats2
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        last = None
        for jit in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                p, dc, _, llf, _ = conditional_stats2(
                    gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'],
                    teX, m=MCOND, test_y=teY, jitter=jit)
                if jit != 1e-6 and log is not None:
                    log(f'    [point_stats] chunk {i//chunk}: jitter '
                        f'escalated to {jit:g} (Cholesky failure at default)')
                break
            except torch._C._LinAlgError as e:
                last = e
        else:
            raise RuntimeError(f'point_stats failed at all jitters: {last}')
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


# --------------------------------------------------------------------------- #
#  hardware policy: chunk-parallel inference for the numpy benchmark models    #
# --------------------------------------------------------------------------- #
_PAR_MODEL = None


def _par_worker(args):
    lo, hi, Wq, Xq = args
    p, dc, ll = _PAR_MODEL.stats(Wq[lo:hi], Xq[lo:hi], chunk=4096)
    return lo, p, dc, ll


def stats_parallel(model, Wq, Xq, n_jobs=None, chunk=4096):
    """model.stats() split over query chunks on a fork pool.  Results are
    identical to the serial call (pure per-row computation).  n_jobs from
    NJOBS (default: half the cores, min 1)."""
    import multiprocessing as mp
    global _PAR_MODEL
    if n_jobs is None:
        n_jobs = int(os.environ.get('NJOBS', max(1, (os.cpu_count() or 4) // 2)))
    n = len(Wq)
    if n_jobs <= 1 or n < 2 * chunk:
        return model.stats(Wq, Xq, chunk=chunk)
    bounds = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    _PAR_MODEL = model
    ctx = mp.get_context('fork')
    with ctx.Pool(n_jobs) as pool:
        parts = pool.map(_par_worker, [(lo, hi, Wq, Xq) for lo, hi in bounds])
    _PAR_MODEL = None
    parts.sort(key=lambda t: t[0])
    return (np.concatenate([t[1] for t in parts]), np.concatenate([t[2] for t in parts]),
            np.concatenate([t[3] for t in parts]))


# ---- chunk-parallel GP inference (spawn pool; torch is not fork-safe) ---- #
_GP_W = None


def _gp_init(gp_path, nthreads):
    global _GP_W
    import torch
    torch.set_num_threads(nthreads)
    try:
        _GP_W = torch.load(gp_path, weights_only=False)
    except TypeError:
        _GP_W = torch.load(gp_path)


def _gp_worker(args):
    lo, Wq, Xq = args
    p, dc, ll = point_stats_robust(_GP_W, Wq, Xq)
    return lo, p, dc, ll


def point_stats_parallel(gp, Wq, Xq, gp_path=None, n_jobs=None, chunk=4096, log=None):
    """point_stats_robust split over query chunks on a spawn pool.  Each
    chunk is processed exactly as the serial call (same jitter ladder), so
    results are identical.  Needs the GP on disk (gp_path); falls back to
    the serial call when gp_path is None or the query is small."""
    import multiprocessing as mp
    if n_jobs is None:
        n_jobs = int(os.environ.get('NJOBS', max(1, (os.cpu_count() or 4) // 2)))
    n = len(Wq)
    main_ok = os.path.isfile(getattr(sys.modules.get('__main__'), '__file__', '') or '')
    if gp_path is None or not os.path.exists(gp_path) or n_jobs <= 1 \
            or n < 2 * chunk or not main_ok:
        return point_stats_robust(gp, Wq, Xq, chunk=chunk, log=log)
    bounds = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_jobs, initializer=_gp_init, initargs=(gp_path, 1)) as pool:
        parts = pool.map(_gp_worker, [(lo, Wq[lo:hi], Xq[lo:hi]) for lo, hi in bounds])
    parts.sort(key=lambda t: t[0])
    return (np.concatenate([t[1] for t in parts]), np.concatenate([t[2] for t in parts]),
            np.concatenate([t[3] for t in parts]))
