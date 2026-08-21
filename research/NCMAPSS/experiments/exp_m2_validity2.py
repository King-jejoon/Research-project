"""
exp_m2_validity.py — SMALL-MODEL validity check for GP-native m2 cleaning.

Question: can the SURPRISE term of the corrected conditional likelihood
(m2 = r^T S^-1 r, ~chi2_q for healthy data) replace the model-free kNN robust
score as the cleaning criterion, keeping the innovation inside the GP
framework?  This is a cheap validity probe, NOT the final experiment:
  * small GP: 2048 training points (1/4 of the deployed 8192)
  * 3 seeds, evaluation sets of 4096
  * model family frozen as always: 5 sensors, RBF, rank 1, Vecchia m=18

Scoring without circularity (2-fold cross-scoring):
  candidates (16384) -> folds F1/F2; a small GP fitted on a 2048-subsample of
  F1 scores every row of F2 (m2 from conditional_stats2 with test_y) and vice
  versa, so no row is scored by a model that saw it.

Swap-paired comparison (same design as exp_clean_proof):
  S        base draw of 2048 candidates
  m2clean  flagged-by-m2 members of S swapped for unflagged donors
  knnclean flagged-by-kNN members of S swapped for unflagged donors
  rand     |m2 flags| random innocents swapped for the same donor pool
Thresholds: m2 > 15.09 (chi2_5, p<0.01) absolute; kNN score > 5 (as proven).
Also reported: overlap between the two flag sets, m2 score distribution.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, predict, true_onset_mask, SIDX, TRAIN_CYC, MCOND, DT
from demo_cond2 import conditional_stats2

NTR = 2048            # small model
NCAND = 16384
NEVAL = 4096
M2_CUT = 15.09        # chi2_5 upper 1% point (absolute)
KNN_CUT = 5.0
SEEDS = [0, 1, 2]
EARLY_HI = 8
RES = os.path.join(HERE, 'm2_validity2_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


@torch.no_grad()
def m2_of(gp, Wq, Xq, chunk=4096):
    m2s, dcs = [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        _, dc, m2, _, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                             gp['tY'], gp['struct'], teX,
                                             m=MCOND, test_y=teY)
        m2s.append(m2); dcs.append(dc)
    return np.concatenate(m2s), np.concatenate(dcs)


def fit_rows(W, X, rows, spare, seed, tries=5):
    """fit on the given rows; on a Cholesky failure swap 20 random members for
    spare rows and retry (identical policy for every configuration)."""
    rows = np.asarray(rows).copy()
    for t in range(tries):
        try:
            return fit_gp(W[rows], X[rows])
        except torch._C._LinAlgError:
            rg = np.random.default_rng(seed * 77 + t)
            out = rg.choice(len(rows), 20, replace=False)
            rows[out] = rg.choice(spare, 20, replace=False)
    raise RuntimeError('fit failed after retries')


def metrics(r, sig):
    z = r / sig
    zn = np.sqrt((z ** 2).mean(1))
    keep = zn <= np.percentile(zn, 90)
    return (float(np.sqrt((z ** 2).mean())),
            float(np.sqrt((z[keep] ** 2).mean())),
            float(np.median(zn)))


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'M2 VALIDITY PROBE (small GP, {NTR} training pts, seeds={SEEDS})')
    log(f'  m2 cut {M2_CUT} (chi2_5 p<0.01, absolute) | kNN cut {KNN_CUT}')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy = true_onset_mask(A)
    early = np.where(healthy & (cyc <= EARLY_HI))[0]
    zk = np.load(os.path.join(HERE, 'raw_clean.npz'))
    assert np.array_equal(zk['pool'], pool)
    knn_score = zk['score']

    R = {}
    for sd in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(40_000 + sd)
        perm = rng.permutation(len(pool))
        cand_i = perm[:NCAND]
        evin = pool[perm[NCAND:NCAND + NEVAL]]
        evearly = rng.choice(early, NEVAL, replace=False)

        # ---- 2-fold m2 scoring ----
        half = NCAND // 2
        F1, F2 = cand_i[:half], cand_i[half:]
        m2s = np.empty(NCAND); dcs = np.empty(NCAND)
        for fa, fb, sl in [(F1, F2, slice(half, NCAND)), (F2, F1, slice(0, half))]:
            pos = rng.choice(half, NTR, replace=False)
            tr = pool[fa[pos]]
            spare = pool[np.setdiff1d(fa, fa[pos], assume_unique=False)]
            gp = fit_rows(W, X, tr, spare, 40_000 + sd)
            m2s[sl], dcs[sl] = m2_of(gp, W[pool[fb]], X[pool[fb]])
        fl_m2 = m2s > M2_CUT
        fl_knn = knn_score[cand_i] > KNN_CUT
        dc_med = float(np.median(dcs))
        fl_2ax = fl_m2 & (dcs >= dc_med)      # surprise WHERE the GP is confident
        both = (fl_m2 & fl_knn).sum()
        log(f'seed{sd}: m2 pct[50,90,99]={np.percentile(m2s,[50,90,99]).round(2)}  '
            f'flags m2={fl_m2.mean():.2%} 2ax={fl_2ax.mean():.2%} knn={fl_knn.mean():.2%}  '
            f'overlap(m2,knn)={both}  dc_med={dc_med:.2f}  ({time.time()-t0:.0f}s)')

        # ---- swap-paired models ----
        base_pos = rng.choice(NCAND, NTR, replace=False)
        base_i = cand_i[base_pos]
        outside = np.setdiff1d(np.arange(NCAND), base_pos)
        donors_ok = outside[~fl_m2[outside] & ~fl_knn[outside]]

        def build(flags):
            f = flags[base_pos]
            n_f = int(f.sum())
            don = donors_ok[rng.choice(len(donors_ok), n_f, replace=False)]
            return np.concatenate([base_i[~f], cand_i[don]]), n_f

        set_m2, n_m2 = build(fl_m2)
        set_2ax, n_2ax = build(fl_2ax)
        set_knn, n_knn = build(fl_knn)
        rr = np.random.default_rng(sd * 991)
        innocent = np.where(~fl_m2[base_pos] & ~fl_knn[base_pos])[0]
        kick = innocent[rr.choice(len(innocent), n_m2, replace=False)]
        mask = np.ones(NTR, bool); mask[kick] = False
        don_r = donors_ok[rr.choice(len(donors_ok), n_m2, replace=False)]
        set_rand = np.concatenate([base_i[mask], cand_i[don_r]])
        log(f'  flags in S: m2={n_m2} 2ax={n_2ax} knn={n_knn}')

        spare_rows = pool[cand_i[donors_ok]]
        sig = {}
        for name, idx in [('base', base_i), ('m2clean', set_m2),
                          ('m2dc', set_2ax), ('knnclean', set_knn),
                          ('rand', set_rand)]:
            t1 = time.time()
            gp = fit_rows(W, X, pool[idx], spare_rows, 50_000 + sd)
            row = {}
            for snm, ev in [('in', evin), ('early', evearly)]:
                r = X[ev] - predict(gp, W[ev])
                if name == 'base':
                    sig[snm] = r.std(0)
                row[snm] = metrics(r, sig[snm])
            R[(sd, name)] = row
            log(f'  {name:>8}: in={row["in"][0]:.4f}/{row["in"][1]:.4f}/{row["in"][2]:.4f}  '
                f'early={row["early"][0]:.4f}/{row["early"][1]:.4f}/{row["early"][2]:.4f}  '
                f'({time.time()-t1:.0f}s)')

    log('')
    labels = ['zRMSE', 'trim', 'med']
    for snm in ['in', 'early']:
        log(f'=== eval_{snm} (mean over seeds; d = rand - X, +는 X 우세) ===')
        for mi, lab in enumerate(labels):
            v = {n: np.array([R[(sd, n)][snm][mi] for sd in SEEDS])
                 for n in ['base', 'm2clean', 'm2dc', 'knnclean', 'rand']}
            out = [f'  {lab:>6}: base {v["base"].mean():.4f}']
            for n, tag in [('m2clean', 'm2'), ('m2dc', 'm2^dc'), ('knnclean', 'knn')]:
                d = v['rand'] - v[n]
                out.append(f'{tag} {v[n].mean():.4f} '
                           f'({int((d>0).sum())}/3, {100*d.mean()/v["rand"].mean():+.2f}%)')
            out.append(f'rand {v["rand"].mean():.4f}')
            log(' | '.join(out))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
