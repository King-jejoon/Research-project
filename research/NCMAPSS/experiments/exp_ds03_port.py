"""
exp_ds03_port.py — FAITHFUL PORT of the colleague's DEMO workflow (developed on
N-CMAPSS DS02-006) to DS03-012, end-to-end through HI + RUL.

Colleague's workflow (Permutation&Grouping-MOGP.ipynb):
  #1  MOGP-Vecchia normal model trained on cycle<5 rows of every dev unit
      (8192 subsampled points, 5 sensors [T30,T48,T50,Nc,Wf], matern32, rank 1)
  #2  state-change detection: per-cycle conditional LL -> mean-drop changepoint
  #3  uncertainty filter: drop points with low detcov before aggregating

Necessary adaptations for DS03-012 (documented, minimal):
  a. LL whitening bug fixed (demo_cond2.conditional_stats2); original quad kept
     out -- shown to break detection (63.9% vs 82.7% acc on DS03).
  b. W inputs = all 4 operating conditions (alt,Mach,TRA,T2). The notebook's
     iloc[:,1:] dropped alt and injected unit-id; not reproduced.
  c. changepoint search range extended from hardcoded k in [5,30) to
     [5, ncycles-5) -- DS03 units live up to 93 cycles with onsets up to 38.
  d. detcov filter threshold percentile-based (--drop), not the DS02-specific
     hardcoded 3.60.

Heavy per-point stats (pred/detcov/m2/llfix) are cached per seed+config in
port_stats_<tag>_s<sd>.npz so filter/detector sweeps re-use them.
Every (pred, true) RUL pair is saved to port_pred_<tag>.npz for post-hoc
metric computation.  Metrics: truncation RMSE + NASA PHM'08 score
  s = sum_i [ exp(-d_i/13)-1 if d_i<0 else exp(d_i/10)-1 ],  d = pred - true.

Run (faithful port, colleague defaults):
  /opt/anaconda3/envs/pt_prac/bin/python3 exp_ds03_port.py
Options: --sensors 5|3  --kernel matern32|rbf  --rank 1|2  --m 18  --npts 8192
         --steps 120  --mcond 15  --drop 0.0  --detect cp|ewma  --tag auto
"""
import os, sys, time, argparse
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
from scipy import stats as sstats
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2
import neural_fusion as NF

DT = torch.float32
SENSORS5 = ['T30', 'T48', 'T50', 'Nc', 'Wf']      # colleague's output_idx [1,2,3,12,13]
SENSORS3 = ['T48', 'T50', 'Wf']                   # our previous set
WARMUP, LAM = 10, 0.2
NPER = 60                                          # overridable via --nper
TRAIN_CYC = 5                                     # colleague: train on cycle < 5
L0, L1, L2, INIT_THR, EPOCHS = 1.0, 6.0, 2.0, 0.2, 1000
FRACS = [0.2, 0.4, 0.6, 0.8]
SEEDS = [0, 1, 2]


def log(s, res):
    print(s, flush=True)
    with open(res, 'a') as f: f.write(s + '\n')


def groups_of(rows, cyc, n_per, seed):
    rng = np.random.default_rng(seed); out = []
    for cc in np.unique(cyc):
        r = rows[cyc == cc]
        if len(r) > n_per: r = rng.choice(r, n_per, replace=False)
        out.append(r)
    return np.unique(cyc), out


def fit_mogp(Wtr, Ytr, kernel, rank, m, steps, lr=0.05):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT); tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=Ytr.shape[1], rank=rank, kernel=kernel).to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Ytr.shape[1]).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=lr, group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


def mean_drop_changepoint(x, kmin=5):
    """Colleague's cell-15 detector, search range extended to the unit length.
    Splits the sequence at k (first k cycles vs rest) and takes the k that
    maximizes the standardized mean-drop statistic."""
    x = np.asarray(x, float); n = len(x)
    kmax = max(kmin + 1, n - 5)
    sigma = np.std(x, ddof=1) + 1e-12
    cs = np.cumsum(x)
    Tst = np.full(n - 1, -np.inf)
    for k in range(kmin, kmax):
        m1 = cs[k - 1] / k
        m2 = (cs[-1] - cs[k - 1]) / (n - k)
        Tst[k - 1] = np.sqrt(k * (n - k) / n) * (m1 - m2) / sigma
    kstar = int(np.argmax(Tst) + 1)       # first k cycles = normal segment
    return kstar, Tst


def true_onset(hs_by_cycle, ucyc):
    below = np.where(hs_by_cycle < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def nasa_score(P, T):
    d = np.asarray(P, float) - np.asarray(T, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rul_eval(devHI, testHI, fracs):
    """Bayesian first-passage truncation RUL. Returns pooled arrays P, T and
    labels (frac, unit) for every prediction."""
    t = np.arange(500).reshape(-1, 1) / 500; Psi = np.hstack((np.ones((500, 1)), t, t**2))
    Yd = [devHI[u] for u in sorted(devHI)]; ntr = len(Yd)
    gam = np.zeros((ntr, 3))
    for i, y in enumerate(Yd):
        n = len(y); Xp = Psi[:n]; gam[i] = (inv(Xp.T@Xp)@Xp.T@y).T
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gam[i])[3:]) ** 2).sum() / (len(Yd[i]) - 4) for i in range(ntr)])
    mu0, cov0 = gam.mean(0), np.cov(gam.T)
    def rul(HI):
        p = Psi[:len(HI)]; mu = inv((p.T@p)/s2 + inv(cov0)) @ ((p.T@HI)/s2 + inv(cov0)@mu0)
        cov = inv((p.T@p)/s2 + inv(cov0)); fx = lambda pp: (pp@mu - 1) / pow(pp@cov@pp.T, 0.5)
        tmin = len(HI)-1; pmin = 0
        for tt in range(len(HI)-1, 500):
            pr = sstats.norm.cdf(fx(Psi[tt]))
            if pr > pmin and pr <= 0.5: pmin = pr; tmin = tt
            if pr > 0.9: break
        tmax = min(tmin+1, len(Psi)-1); pmax = sstats.norm.cdf(fx(Psi[tmax]))
        return (tmax if pmax == pmin else tmax - (tmax-tmin)*(pmax-0.5)/(pmax-pmin)) - len(HI) + 1
    P, T, FR, UN = [], [], [], []
    for f in fracs:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n * f))
            P.append(rul(h[:cut])); T.append(n - cut); FR.append(f); UN.append(u)
    return np.array(P), np.array(T), np.array(FR), np.array(UN)


def compute_stats(cache, sd, cfg, cache_path):
    """Fit normal MOGP on cycle<TRAIN_CYC and compute per-point conditional
    stats for every dev+test unit.  Cached to npz."""
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        return z
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    Wt, Xtall, At = cache['W_test'], cache['X_s_test'], cache['A_test']
    sidx = cfg['sidx']
    X, Xt = Xall[:, sidx], Xtall[:, sidx]

    # ---- #1 training set: cycle < TRAIN_CYC on all dev units, npts subsample ----
    cycles = A[:, 1].astype(int)
    pool = np.where(cycles < TRAIN_CYC)[0]
    rng = np.random.default_rng(sd)
    tr = rng.choice(pool, min(cfg['npts'], len(pool)), replace=False)
    gp = fit_mogp(W[tr], X[tr], cfg['kernel'], cfg['rank'], cfg['m'], cfg['steps'], lr=cfg.get('lr', 0.05))

    out = {}
    for split, (Wd, Xd, Ad) in [('dev', (W, X, A)), ('test', (Wt, Xt, At))]:
        units = np.unique(Ad[:, 0].astype(int))
        for u in units:
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + sd)
            idx = np.concatenate(sel)
            cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
            teX = torch.tensor(gp['xs'].transform(Wd[idx]), dtype=DT)
            teY = torch.tensor(gp['ys'].transform(Xd[idx]), dtype=DT)
            pred_s, detcov, m2, llfix, _ = conditional_stats2(
                gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'], teX,
                m=cfg['mcond'], test_y=teY)
            pred = pred_s * gp['ys'].scale_ + gp['ys'].mean_       # de-standardize
            resid = Xd[idx] - pred
            key = f'{split}_{u}'
            out[f'{key}_cc'] = cc.astype(np.int32)
            out[f'{key}_resid'] = resid.astype(np.float32)
            out[f'{key}_detcov'] = detcov.astype(np.float32)
            out[f'{key}_m2'] = m2.astype(np.float32)
            out[f'{key}_llfix'] = llfix.astype(np.float32)
            if split == 'dev':
                hs = np.array([Ad[rows[cyc == c0], 3].mean() for c0 in ucyc])
                out[f'{key}_hs'] = hs.astype(np.float32)
    np.savez_compressed(cache_path, **out)
    return np.load(cache_path, allow_pickle=False)


def per_cycle(z, key, arr_name, agg='mean'):
    cc = z[f'{key}_cc']; v = z[f'{key}_{arr_name}']
    ucyc = np.unique(cc)
    if v.ndim == 1:
        return ucyc, np.array([v[cc == k].mean() for k in ucyc])
    return ucyc, np.stack([v[cc == k].mean(0) for k in ucyc])


def run_seed(sd, cache, cfg, res):
    torch.manual_seed(sd); np.random.seed(sd)
    cache_path = os.path.join(HERE, f"port_stats_{cfg['tag']}_s{sd}.npz")
    t0 = time.time()
    z = compute_stats(cache, sd, cfg, cache_path)
    log(f'  seed{sd}: stats ready ({time.time()-t0:.0f}s)', res)

    du = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('dev_') and k.endswith('_cc')})
    tu = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('test_') and k.endswith('_cc')})

    # ---- #2 detection on dev (scored vs hs) ----
    recs = []
    for u in du:
        key = f'dev_{u}'
        ucyc, ll_c = per_cycle(z, key, 'llfix')
        _, m2_c = per_cycle(z, key, 'm2')
        tru = true_onset(z[f'{key}_hs'], ucyc)
        if cfg['detect'] == 'cp':
            kstar, _ = mean_drop_changepoint(ll_c)
            trans = int(ucyc[kstar]) if kstar < len(ucyc) else int(ucyc[-1] + 1)
        else:                                        # ewma on m2, K=3
            ic = m2_c[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
            UCL = mu0 + 3.0 * sd0 * np.sqrt(LAM / (2 - LAM))
            e = mu0; E = []
            for v in m2_c: e = LAM * v + (1 - LAM) * e; E.append(e)
            cr = np.where(np.array(E) > UCL)[0]
            trans = int(ucyc[cr[0]]) if len(cr) else int(ucyc[-1] + 1)
        pred_lab = (ucyc >= trans).astype(int); true_lab = (ucyc >= tru).astype(int)
        acc = float((pred_lab == true_lab).mean())
        recs.append(dict(unit=u, true=tru, pred=trans, delay=trans - tru, acc=acc))

    # ---- #3 residual aggregation with detcov percentile filter ----
    sens = cfg['sens']
    def resid_dict(units, split):
        out = {}
        for u in units:
            key = f'{split}_{u}'
            cc = z[f'{key}_cc']; r = z[f'{key}_resid']; dc = z[f'{key}_detcov']
            keepmask = np.ones(len(dc), bool)
            if cfg['drop'] > 0:
                thr = np.percentile(dc, cfg['drop'] * 100)
                keepmask = dc >= thr                 # keep high detcov = low uncertainty
            ucyc = np.unique(cc)
            d = {s: {} for s in sens}
            for k in ucyc:
                mk = (cc == k) & keepmask
                if mk.sum() == 0: mk = cc == k
                mv = r[mk].mean(0)
                for j, s in enumerate(sens): d[s][int(k)] = float(mv[j])
            out[u] = d
        return out
    dev_res = resid_dict(du, 'dev'); test_res = resid_dict(tu, 'test')
    st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
              np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in sens}
    nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in sens} for u in dd}
    dev_n, test_n = nrm(dev_res), nrm(test_res)

    # ---- HI + RUL ----
    def blist(dn):
        out = []
        for u in sorted(dn):
            cy = sorted(dn[u][sens[0]].keys())
            out.append((u, cy, np.array([[dn[u][s][k] for s in sens] for k in cy])))
        return out
    dl, tl = blist(dev_n), blist(test_n)
    np.random.seed(sd)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=EPOCHS, lambda0=L0, lambda1=L1,
                            lambda2=L2, init_threshold=INIT_THR, alpha=0.001, verbose=False)
    dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
    tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
    P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
    return P, T, FR, UN, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sensors', type=int, default=5, choices=[3, 5])
    ap.add_argument('--kernel', default='matern32', choices=['matern32', 'rbf'])
    ap.add_argument('--rank', type=int, default=1)
    ap.add_argument('--m', type=int, default=18)
    ap.add_argument('--npts', type=int, default=8192)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--mcond', type=int, default=15)
    ap.add_argument('--drop', type=float, default=0.0)
    ap.add_argument('--detect', default='cp', choices=['cp', 'ewma'])
    ap.add_argument('--nper', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    global NPER
    NPER = args.nper
    sens = SENSORS5 if args.sensors == 5 else SENSORS3
    sidx = [L.OUTPUT_NAMES.index(s) for s in sens]
    tag = args.tag or (f's{args.sensors}_{args.kernel}_r{args.rank}_m{args.m}_n{args.npts}'
                       + (f'_p{args.nper}' if args.nper != 60 else '')
                       + (f'_lr{args.lr}' if args.lr != 0.05 else ''))
    cfg = dict(sens=sens, sidx=sidx, kernel=args.kernel, rank=args.rank, m=args.m,
               npts=args.npts, steps=args.steps, mcond=args.mcond,
               drop=args.drop, detect=args.detect, tag=tag, lr=args.lr)
    res = os.path.join(HERE, f'port_{tag}_drop{int(args.drop*100)}_{args.detect}_results.txt')
    open(res, 'w').close()
    log(f'DS03-012 PORT of colleague workflow  tag={tag}  drop={args.drop}  detect={args.detect}', res)
    log(f'  sensors={sens}  kernel={args.kernel} rank={args.rank} m={args.m} '
        f'npts={args.npts} steps={args.steps} mcond={args.mcond}  train=cycle<{TRAIN_CYC}', res)

    cache = L.load_cache(); t0 = time.time()
    allP, allT, allFR = [], [], []
    all_recs = []
    for sd in SEEDS:
        P, T, FR, UN, recs = run_seed(sd, cache, cfg, res)
        allP.append(P); allT.append(T); allFR.append(FR); all_recs.append(recs)
        a = np.mean([r['acc'] for r in recs]) * 100
        rmse = float(np.sqrt(((P - T) ** 2).mean())); sc = nasa_score(P, T)
        log(f'  seed{sd}: det acc={a:.1f}%  trans={[r["pred"] for r in recs]}  '
            f'RUL RMSE={rmse:.2f}  score={sc:.1f}  ({time.time()-t0:.0f}s)', res)
    np.savez(os.path.join(HERE, f'port_pred_{tag}_drop{int(args.drop*100)}_{args.detect}.npz'),
             P=np.stack(allP), T=np.stack(allT), FR=np.stack(allFR), seeds=np.array(SEEDS))

    # ---- report ----
    flat = [r for recs in all_recs for r in recs]
    log('\n' + '=' * 76, res)
    log(f'DETECTION ({args.detect}, fixed LL): acc = {np.mean([r["acc"] for r in flat])*100:.1f}%  '
        f'mean |delay| = {np.mean([abs(r["delay"]) for r in flat]):.1f} cyc', res)
    log('=' * 76, res)
    log(f'{"data used":>10} | {"RMSE s0/s1/s2":>21} | {"RMSE mean":>9} | {"score mean":>10}', res)
    log('-' * 76, res)
    for f in FRACS:
        rs, scs = [], []
        for P, T, FR in zip(allP, allT, allFR):
            mk = FR == f
            rs.append(float(np.sqrt(((P[mk] - T[mk]) ** 2).mean())))
            scs.append(nasa_score(P[mk], T[mk]))
        log(f'{f:>9.0%} | {rs[0]:>6.2f} {rs[1]:>6.2f} {rs[2]:>6.2f} | {np.mean(rs):>9.2f} | {np.mean(scs):>10.1f}', res)
    rmses = np.array([float(np.sqrt(((P - T) ** 2).mean())) for P, T in zip(allP, allT)])
    scores = np.array([nasa_score(P, T) for P, T in zip(allP, allT)])
    log('-' * 76, res)
    log(f'{"OVERALL":>10} | {rmses[0]:>6.2f} {rmses[1]:>6.2f} {rmses[2]:>6.2f} | '
        f'{rmses.mean():>9.2f} | {scores.mean():>10.1f}', res)
    log(f'OVERALL RUL RMSE = {rmses.mean():.2f} +/- {rmses.std():.2f}   '
        f'NASA score = {scores.mean():.1f} +/- {scores.std():.1f}  (n=24 preds/seed)', res)
    log(f'wall time = {time.time()-t0:.0f}s', res)


if __name__ == '__main__':
    main()
