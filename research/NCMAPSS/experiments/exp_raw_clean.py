"""
exp_raw_clean.py — PHASE 1: RAW-DATA CLEANING (remove unexplainable measurements)

What is cleaned
  The raw cycle<5 training pool itself.  For every pool row a MODEL-FREE robust
  score is computed from its k=20 nearest neighbours in standardized W space:
        rz_s = (x_s - median_nb(x_s)) / (1.4826 * MAD_nb(x_s)),  score = RMS_s(rz_s)
  A row whose sensors disagree with other engines at virtually the same
  operating condition is an unexplainable measurement.  Rows with score > c are
  deleted from the pool; the 8192 training points are then drawn from the
  cleaned pool.  c is an ABSOLUTE, transferable threshold (robust-z units) and
  the score never touches the GP, so there is no circularity.

Controls (both mandatory, learned from earlier failures)
  * baseline   : no cleaning
  * random     : delete the SAME number of rows at random.  Any gain that
                 survives this control is attributable to the criterion, not to
                 resampling/retraining.

Evaluation — fixed, COMMON, and deliberately NOT cleaned, so no selection effect
  eval_in    8192 held-out cycle<5 rows            (pure healthy, in-distribution)
  eval_early 8192 healthy rows from cycles 5..8    (unseen cycles, minimal drift)
  eval_gen   8192 healthy pre-onset rows           (legacy metric; CONFOUNDED —
             Phase 0c showed |z| grows with cycle in every unit, i.e. this set
             contains real degradation, so it is reported but not used to select)
  eval_deg   8192 post-onset rows                  (for the separation ratio)
Metrics per set: zRMSE, trimmed zRMSE (central 90 %), median |z|; plus
  separation = zRMS(eval_deg) / zRMS(eval_early)   -- higher is better for
  detection, and it is the property the downstream pipeline actually needs.
Per-sensor sigma is frozen from the BASELINE model on each set.

Model is frozen throughout: 5 sensors, RBF, rank 1, Vecchia m=18, Adam
lr 0.1 x 120, StandardScaler.  3 seeds.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, predict, true_onset_mask, SIDX, NPTS, TRAIN_CYC

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
NEVAL = 8192
K = 20
CUTS = [float(x) for x in os.environ.get('CUTS', '5,3,2').split(',')]
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]
EARLY_HI = 8                      # eval_early uses cycles TRAIN_CYC..EARLY_HI
RES = os.path.join(HERE, 'raw_clean_results.txt')
NPZ = os.path.join(HERE, 'raw_clean.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def raw_scores(W, X, pool):
    """model-free robust neighbour score for every pool row."""
    Ws = StandardScaler().fit_transform(W[pool])
    Xp = X[pool]
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Ws)
    out = np.empty(len(pool), np.float32)
    for i in range(0, len(pool), 50000):
        _, idx = nn.kneighbors(Ws[i:i + 50000])
        nb = idx[:, 1:]
        Xn = Xp[nb]
        med = np.median(Xn, axis=1)
        mad = np.maximum(np.median(np.abs(Xn - med[:, None, :]), axis=1) * 1.4826, 1e-9)
        rz = (Xp[i:i + 50000] - med) / mad
        out[i:i + 50000] = np.sqrt((rz ** 2).mean(1))
    return out


def fit_robust(W, X, allowed, seed, tries=6):
    """Two-stage sampling exactly as in the frozen pipeline (16384 candidates ->
    8192 training points).  Vecchia Cholesky can fail when a draw contains
    near-duplicate operating points; on failure we redraw with a new sub-seed.
    Identical procedure for every configuration, so no configuration is
    advantaged.  Returns (gp, n_retries)."""
    for t in range(tries):
        rg = np.random.default_rng(seed * 100 + t)
        cand = allowed[rg.choice(len(allowed), min(2 * NPTS, len(allowed)),
                                 replace=False)]
        tr = cand[rg.choice(len(cand), NPTS, replace=False)]
        try:
            return fit_gp(W[tr], X[tr]), t
        except torch._C._LinAlgError:
            continue
    raise RuntimeError(f'GP fit failed after {tries} redraws (seed {seed})')


def stats(r, sig):
    z = r / sig
    zn = np.sqrt((z ** 2).mean(1))
    keep = zn <= np.percentile(zn, 90)
    return dict(zrmse=float(np.sqrt((z ** 2).mean())),
                ztrim=float(np.sqrt((z[keep] ** 2).mean())),
                zmed=float(np.median(zn)))


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('PHASE 1 — RAW-DATA CLEANING (model-free robust neighbour score)')
    log(f'  absolute cuts (score > c): {CUTS}   seeds={SEEDS}   k={K}')
    log('  every cut is paired with a RANDOM removal of the same size (control)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy = true_onset_mask(A)
    early = np.where(healthy & (cyc <= EARLY_HI))[0]
    gen = np.where(healthy)[0]
    # degraded = post-onset rows
    deg_mask = np.zeros(len(A), bool)
    unit = A[:, 0].astype(int)
    for u in np.unique(unit):
        ur = np.where(unit == u)[0]
        uc = np.unique(cyc[ur]); hs = A[:, 3]
        hsb = np.array([hs[ur[cyc[ur] == c]].mean() for c in uc])
        below = np.where(hsb < 0.5)[0]
        if len(below):
            deg_mask[ur] = cyc[ur] >= int(uc[below[0]])
    deg = np.where(deg_mask)[0]
    log(f'  pool={len(pool)}  early(cyc<={EARLY_HI})={len(early)}  '
        f'healthy={len(gen)}  degraded={len(deg)}')

    t0 = time.time()
    score = raw_scores(W, X, pool)
    log(f'  raw scores computed for the whole pool ({time.time()-t0:.0f}s)  '
        f'p50={np.percentile(score,50):.2f} p99={np.percentile(score,99):.2f}')
    for c in CUTS:
        log(f'    cut score>{c}: removes {100*(score>c).mean():.2f}% of the pool')

    R = {}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        evin = rng.choice(pool, NEVAL, replace=False)
        evearly = rng.choice(early, NEVAL, replace=False)
        evgen = rng.choice(gen, NEVAL, replace=False)
        evdeg = rng.choice(deg, NEVAL, replace=False)
        sets = [('in', evin), ('early', evearly), ('gen', evgen), ('deg', evdeg)]
        trainable = np.setdiff1d(pool, evin, assume_unique=False)

        configs = [('baseline', np.ones(len(pool), bool))]
        for c in CUTS:
            keep = score <= c
            configs.append((f'clean>{c:g}', keep))
            nrm = int((~keep).sum())
            rr = np.random.default_rng(sd * 1000 + int(c * 10))
            rmask = np.ones(len(pool), bool)
            rmask[rr.choice(len(pool), nrm, replace=False)] = False
            configs.append((f'random{100*nrm/len(pool):.1f}%', rmask))

        sig = {}
        for ci, (name, keepmask) in enumerate(configs):
            t0 = time.time()
            allowed = pool[keepmask]
            allowed = np.setdiff1d(allowed, evin, assume_unique=False)
            gp, nret = fit_robust(W, X, allowed, sd * 100 + ci)
            row = {}
            for snm, ev in sets:
                r = X[ev] - predict(gp, W[ev])
                if name == 'baseline':
                    sig[snm] = r.std(0)
                row[snm] = stats(r, sig[snm])
            row['sep'] = row['deg']['zrmse'] / row['early']['zrmse']
            R[(sd, name)] = row
            log(f'  seed{sd} {name:>12}: [redraws={nret}] in={row["in"]["zrmse"]:.4f} '
                f'early={row["early"]["zrmse"]:.4f} '
                f'(trim {row["early"]["ztrim"]:.4f}, med {row["early"]["zmed"]:.4f}) '
                f'gen={row["gen"]["zrmse"]:.4f}  sep={row["sep"]:.3f}  '
                f'({time.time()-t0:.0f}s)')

    names = [n for n, _ in configs]
    log('')
    log(f'{"config":>12} | {"eval_in":>15} | {"eval_early":>15} | '
        f'{"early trim":>15} | {"eval_gen":>15} | {"separation":>13}')
    log('-' * 100)
    for n in names:
        f = lambda k, s='zrmse': np.array([R[(sd, n)][k][s] if s else R[(sd, n)][k]
                                           for sd in SEEDS])
        sep = np.array([R[(sd, n)]['sep'] for sd in SEEDS])
        log(f'{n:>12} | {f("in").mean():7.4f}±{f("in").std():<6.4f} | '
            f'{f("early").mean():7.4f}±{f("early").std():<6.4f} | '
            f'{f("early","ztrim").mean():7.4f}±{f("early","ztrim").std():<6.4f} | '
            f'{f("gen").mean():7.4f}±{f("gen").std():<6.4f} | '
            f'{sep.mean():6.3f}±{sep.std():<5.3f}')
    np.savez(NPZ, score=score, pool=pool,
             **{f'{sd}_{n}_{k}': np.array([R[(sd, n)][k][s] for s in
                                           ('zrmse', 'ztrim', 'zmed')])
                for sd in SEEDS for n in names for k in ('in', 'early', 'gen', 'deg')})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
