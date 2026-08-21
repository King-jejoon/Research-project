"""
exp_clean_proof.py — DEFINITIVE TEST that raw-data cleaning improves the model.

The earlier sweep compared a cleaned pool against a randomly thinned pool, but
each configuration drew its own 8192 training points, so the draw-to-draw
variance (+-0.1 zRMSE) swamped the effect.  Here the contrast is a SWAP that
holds the draw fixed:

  S        base draw of 8192 points from the 16384 candidates
  f        the flagged members of S            (model-free score > CUT)
  r        an equal number of UNflagged members of S, chosen at random
  Donors   n_f fresh unflagged candidates outside S  -- the SAME set for both

  M_base   trained on S
  M_clean  trained on (S \ f) + Donors      <- flagged points swapped out
  M_rand   trained on (S \ r) + Donors      <- innocent points swapped out

M_clean and M_rand contain the same number of points, receive the identical
donor points, and share ~96 % of their training set.  They differ ONLY in
which 3.7 % were evicted, so any difference is attributable to the criterion
and nothing else.

Evaluation: fixed sets per seed, never cleaned, common to all three models
  eval_in     8192 held-out cycle<5 rows
  eval_early  8192 healthy rows from cycles 5..8
Metrics: zRMSE, trimmed zRMSE (central 90 %), median |z|; per-sensor sigma
frozen from M_base.

Mechanism check: every eval point is labelled with its distance to the nearest
EVICTED point in standardized W space.  If cleaning works causally, the gain of
M_clean over M_rand must concentrate where the flagged points used to be.

Model frozen throughout: 5 sensors, RBF, rank 1, Vecchia m=18, Adam lr 0.1x120.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, predict, true_onset_mask, SIDX, NPTS, NCAND, TRAIN_CYC

NEVAL = 8192
CUT = float(os.environ.get('CUT', 5.0))
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2,3,4,5,6,7').split(',')]
EARLY_HI = 8
RES = os.path.join(HERE, 'clean_proof_results.txt')
NPZ = os.path.join(HERE, 'clean_proof.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def metrics(r, sig):
    z = r / sig
    zn = np.sqrt((z ** 2).mean(1))
    keep = zn <= np.percentile(zn, 90)
    return (float(np.sqrt((z ** 2).mean())),
            float(np.sqrt((z[keep] ** 2).mean())),
            float(np.median(zn))), zn


def fit_try(W, X, tr, seed, donors_pool, n_swap, tries=5):
    """fit; on a Cholesky failure redraw the donor points and retry."""
    for t in range(tries):
        try:
            return fit_gp(W[tr], X[tr]), t
        except torch._C._LinAlgError:
            rg = np.random.default_rng(seed * 31 + t + 1)
            tr = np.concatenate([tr[:-n_swap],
                                 donors_pool[rg.choice(len(donors_pool), n_swap,
                                                       replace=False)]])
    raise RuntimeError('GP fit failed after redraws')


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('DEFINITIVE TEST — swap-paired comparison of raw-data cleaning')
    log(f'  cut: model-free score > {CUT}   seeds={SEEDS}')
    log('  M_clean and M_rand share ~96% of their training points and get the '
        'same donors; only the eviction rule differs.')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy = true_onset_mask(A)
    early = np.where(healthy & (cyc <= EARLY_HI))[0]
    zc = np.load(os.path.join(HERE, 'raw_clean.npz'))
    assert np.array_equal(zc['pool'], pool)
    score = zc['score']
    flagged_pool = score > CUT
    log(f'  pool={len(pool)}  flagged={100*flagged_pool.mean():.2f}%')

    R, MECH = {}, []
    for sd in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(10_000 + sd)
        perm = rng.permutation(len(pool))
        cand_i = perm[:NCAND]                       # positions inside `pool`
        evin = pool[perm[NCAND:NCAND + NEVAL]]
        evearly = rng.choice(early, NEVAL, replace=False)

        base_i = cand_i[rng.choice(NCAND, NPTS, replace=False)]
        fl = flagged_pool[base_i]
        n_f = int(fl.sum())
        if n_f < 20:
            log(f'seed{sd}: only {n_f} flagged in the draw, skipping'); continue
        outside = np.setdiff1d(cand_i, base_i, assume_unique=False)
        donors_pool = outside[~flagged_pool[outside]]
        donors = donors_pool[rng.choice(len(donors_pool), n_f, replace=False)]

        keep_clean = base_i[~fl]
        innocent = base_i[~fl]
        r_idx = rng.choice(len(innocent), n_f, replace=False)
        mask_r = np.ones(len(base_i), bool)
        mask_r[np.where(~fl)[0][r_idx]] = False
        keep_rand = base_i[mask_r]

        sets = {'base': pool[base_i],
                'clean': pool[np.concatenate([keep_clean, donors])],
                'rand': pool[np.concatenate([keep_rand, donors])]}
        evicted = pool[base_i[fl]]
        log(f'seed{sd}: |S|={len(base_i)}  flagged in S={n_f} '
            f'({100*n_f/NPTS:.2f}%)  donors={len(donors)}  '
            f'overlap(clean,rand)={len(np.intersect1d(sets["clean"],sets["rand"]))}')

        sig = {}
        zmap = {}
        for name in ['base', 'clean', 'rand']:
            gp, nret = fit_try(W, X, sets[name], sd, pool[donors_pool], n_f)
            row = {}
            for snm, ev in [('in', evin), ('early', evearly)]:
                r = X[ev] - predict(gp, W[ev])
                if name == 'base':
                    sig[snm] = r.std(0)
                m, zn = metrics(r, sig[snm])
                row[snm] = m
                if snm == 'early':
                    zmap[name] = zn
            R[(sd, name)] = row
            log(f'  {name:>5}: in(zrmse/trim/med)='
                f'{row["in"][0]:.4f}/{row["in"][1]:.4f}/{row["in"][2]:.4f}  '
                f'early={row["early"][0]:.4f}/{row["early"][1]:.4f}/{row["early"][2]:.4f}'
                f'  [redraws={nret}] ({time.time()-t0:.0f}s)')

        # mechanism: distance from each eval_early point to the nearest evicted point
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(W[sets['base']])
        nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(W[evicted]))
        d, _ = nn.kneighbors(sc.transform(W[evearly]))
        MECH.append((d[:, 0], zmap['clean'], zmap['rand']))

    # ---------- summary ----------
    log('')
    names = ['base', 'clean', 'rand']
    labels = ['zRMSE', 'trimmed zRMSE', 'median |z|']
    for si, snm in enumerate(['in', 'early']):
        log(f'=== eval_{snm} ===')
        for mi, lab in enumerate(labels):
            vals = {n: np.array([R[(sd, n)][snm][mi] for sd in SEEDS
                                 if (sd, n) in R]) for n in names}
            d = vals['rand'] - vals['clean']          # >0 : cleaning better
            wins = int((d > 0).sum()); n = len(d)
            from scipy import stats as st
            p = st.binomtest(wins, n, 0.5).pvalue
            pw = st.wilcoxon(vals['clean'], vals['rand']).pvalue if n >= 6 else np.nan
            log(f'  {lab:>14}: base {vals["base"].mean():.4f} | '
                f'clean {vals["clean"].mean():.4f} | rand {vals["rand"].mean():.4f} '
                f'| clean-vs-rand wins {wins}/{n}  sign p={p:.4f}  '
                f'Wilcoxon p={pw:.4f}  gain {100*d.mean()/vals["rand"].mean():+.2f}%')
        log('')

    log('mechanism — gain of clean over rand by distance to the nearest evicted point:')
    dd = np.concatenate([m[0] for m in MECH])
    zc_ = np.concatenate([m[1] for m in MECH])
    zr_ = np.concatenate([m[2] for m in MECH])
    e = np.percentile(dd, [0, 20, 40, 60, 80, 100])
    log(f'  {"distance bin":>26} | {"n":>7} | {"clean":>7} | {"rand":>7} | {"gain":>7}')
    for i in range(5):
        m = (dd >= e[i]) & (dd <= e[i + 1] if i == 4 else dd < e[i + 1])
        gc, gr = np.median(zc_[m]), np.median(zr_[m])
        log(f'  Q{i+1} [{e[i]:.3f},{e[i+1]:.3f}) | {int(m.sum()):>7} | {gc:7.4f} | '
            f'{gr:7.4f} | {100*(gr-gc)/gr:+6.2f}%')
    np.savez(NPZ, **{f'{sd}_{n}_{s}': np.array(R[(sd, n)][s])
                     for (sd, n) in R for s in ('in', 'early')})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
