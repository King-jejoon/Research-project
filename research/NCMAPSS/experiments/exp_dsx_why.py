"""exp_dsx_why.py — WHY does the DS03 chain not transfer?  Dev-only diagnostics
per sub-dataset (no test contact), same yardsticks everywhere:
 (A) gate precondition: are low-confidence points actually noisy?  Within
     healthy cycles, |z| residual (max over channels, z by healthy sigma) of
     the bottom-30 % detcov points vs the rest; Spearman rho(detcov, |z|);
     detcov spread P10/P50/P90; fraction of points below the frozen V.
 (B) label-signal alignment: oracle onset = first cycle with |z|>3 on any
     channel for 3 consecutive cycles; lag = oracle - hs label, per unit.
 (C) sensor adequacy: model-free kNN (k=10) zRMS screen of all 14 sensors
     (exp_v4_t1_screen protocol, cycle<=3 reference) -> ranking; overlap of
     the frozen set [T30,T48,T50,Nc,Wf] with the dataset's own top-5 and the
     zRMS mass it captures.
 (D) fleet structure: life CV, onset-fraction spread, dev/test class mix.
Usage: DS=ds03|ds01|... python3 exp_dsx_why.py   -> dsx_why_{DS}.txt"""
import os, sys, json
import numpy as np
from scipy import stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exp_lib as L
from exp_v4_hi import trim25
from exp_dsx_lib import load_stats, nbase_at
DS = os.environ['DS']
SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
RES = os.path.join(HERE, f'dsx_why_{DS}.txt'); open(RES, 'w').close()
def log(s):
    print(s, flush=True); open(RES, 'a').write(s + '\n')

# ---- stats + rule ----
if DS == 'ds03':
    from exp_v4_setcmp import load as load_ds03
    def stats(sd):
        D = load_ds03(5, sd)
        return {int(u): dict(cc=d['cc'], dc=d['dc'], ll=d['ll'], resid=None, ucyc=np.asarray(d['ucyc'], int),
                             dur=np.asarray(d['dur'], float), onset=int(d['onset'])) for u, d in D.items()}
    Zs = np.load(os.path.join(HERE, 'v4_c3_stats_s0.npz'))
    S0 = stats(0)
    for u in S0: S0[u]['resid'] = Zs[f'u{u}_resid']
    rule = dict(NB=10, V_sparse=19.0, V_dense=20.2, wval=30.0, wmode='fh')
    cache = L.load_cache()
else:
    S0 = load_stats(DS, 0)
    Z = np.load(os.path.join(HERE, f'{DS}v6_detsel.npz'))
    rule = dict(NB=int(Z['NB'][0]), V_sparse=float(Z['V_sparse'][0]), V_dense=float(Z['V_dense'][0]),
                wval=float(Z['wval'][0]), wmode=str(Z['wmode'][0]))
    cache = np.load(os.path.join(HERE, f'cache_{DS}.npz'))
units = sorted(S0)
log(f'=== {DS.upper()}  rule {rule}')

# ---- (A) gate precondition ----
log('(A) gate precondition — healthy cycles only, seed 0')
all_dc = np.concatenate([S0[u]['dc'] for u in units])
p10, p50, p90 = np.percentile(all_dc, [10, 50, 90])
log(f'    detcov P10/P50/P90 = {p10:.2f}/{p50:.2f}/{p90:.2f}  spread P90-P10 = {p90-p10:.2f}')
rows = []
for u in units:
    d = S0[u]; h = d['cc'] < d['onset']
    if d['resid'] is None or h.sum() < 100: continue
    R = d['resid'][h]; dc = d['dc'][h]
    sig = R.std(0) + 1e-9
    z = np.abs(R / sig).max(1)
    lo = dc <= np.percentile(dc, 30)
    rho = st.spearmanr(dc, z).correlation
    V = rule['V_sparse'] if nbase_at(d, rule['wval'], rule['wmode']) < rule['NB'] else rule['V_dense']
    frac_below = float((d['dc'] < V).mean())
    rows.append((u, z[lo].mean() / z[~lo].mean(), rho, frac_below))
    log(f'    u{u:<3d} |z| low-conf/rest = {z[lo].mean()/z[~lo].mean():.2f}   rho(detcov,|z|) = {rho:+.2f}   below V: {frac_below*100:4.1f}%')
a = np.array([[r[1], r[2], r[3]] for r in rows])
log(f'    MEAN: low-conf points are {a[:,0].mean():.2f}x noisier; rho = {a[:,1].mean():+.2f}; {a[:,2].mean()*100:.1f}% dropped')

# ---- (B) label-signal alignment ----
log('(B) label-signal alignment (oracle = |z|>3 on any channel, 3 consecutive cycles)')
lags = []
for u in units:
    d = S0[u]
    if d['resid'] is None: continue
    uc = np.unique(d['cc'])
    T = np.stack([trim25(d['resid'][d['cc'] == c]) for c in uc])
    h = uc < d['onset']
    mu, sg = T[h].mean(0), T[h].std(0) + 1e-9
    zmax = np.abs((T - mu) / sg).max(1)
    hit = np.where(np.convolve((zmax > 3).astype(int), np.ones(3, int), 'valid') == 3)[0]
    orc = int(uc[hit[0]]) if len(hit) else int(uc[-1] + 1)
    i10 = min(np.searchsorted(uc, d['onset'] + 10), len(uc) - 1)
    lags.append(orc - d['onset'])
    log(f'    u{u:<3d} hs {d["onset"]:3d}  oracle {orc:3d}  lag {orc-d["onset"]:+4d}   |z|max at hs+10 = {zmax[i10]:.1f}   EOL |z|max = {zmax[-1]:.0f}')
lags = np.array(lags)
log(f'    lag mean {lags.mean():+.1f} cycles (median {np.median(lags):+.0f}); units with lag > 10: {int((lags > 10).sum())}/{len(lags)}')

# ---- (C) sensor adequacy: kNN zRMS screen on all 14 sensors ----
log('(C) sensor screen, model-free kNN k=10 (cycle<=3 reference), all 14 sensors')
W, X, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
cyc = A[:, 1].astype(int); hs = A[:, 3]
rng = np.random.default_rng(0)
pool = np.where(cyc <= 3)[0]; deg = np.where(hs < 0.5)[0]; hout = np.where((hs >= 0.5) & (cyc > 3))[0]
ref = rng.choice(pool, min(150000, len(pool)), replace=False)
hold = rng.choice(hout, min(30000, len(hout)), replace=False)
dgs = rng.choice(deg, min(60000, len(deg)), replace=False)
sc = StandardScaler().fit(W[ref]); nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(W[ref]))
def knn_resid(idx):
    _, nb = nn.kneighbors(sc.transform(W[idx]))
    return X[idx] - X[ref][nb].mean(1)
sig = knn_resid(hold).std(0) + 1e-9
zr = np.sqrt((knn_resid(dgs) ** 2).mean(0)) / sig
order = np.argsort(-zr)
names = L.OUTPUT_NAMES
log('    ' + '  '.join(f'{names[j]}={zr[j]:.2f}' for j in order))
top5 = [names[j] for j in order[:5]]
frozen_rank = {s: int(np.where(order == names.index(s))[0][0]) + 1 for s in SENS}
log(f'    top-5 here: {top5}; frozen set ranks: {frozen_rank}; overlap {len(set(top5) & set(SENS))}/5; '
    f'zRMS mass frozen {sum(zr[names.index(s)] for s in SENS):.2f} vs top-5 {zr[order[:5]].sum():.2f}')

# ---- (D) fleet structure ----
Ad = cache['A_dev']; At = cache['A_test']
def fleet(Aa):
    u = Aa[:, 0].astype(int); c = Aa[:, 1].astype(int); fc = Aa[:, 2].astype(int); h = Aa[:, 3]
    lives, ons, cls = [], [], []
    for uu in np.unique(u):
        m = u == uu; lives.append(c[m].max()); cls.append(int(np.bincount(fc[m]).argmax()))
        uc = np.unique(c[m]); hb = np.array([h[m][c[m] == x].mean() for x in uc]); b = np.where(hb < 0.5)[0]
        ons.append(uc[b[0]] / c[m].max() if len(b) else 1.0)
    return np.array(lives), np.array(ons), cls
ld, od, cd = fleet(Ad); lt, ot, ct = fleet(At)
log('(D) fleet structure')
log(f'    dev: n={len(ld)} life {ld.mean():.0f}±{ld.std():.0f} (CV {ld.std()/ld.mean():.2f}) onset-frac {od.mean():.2f}±{od.std():.2f} classes {sorted(cd)}')
log(f'    test: n={len(lt)} life {lt.mean():.0f}±{lt.std():.0f} onset-frac {ot.mean():.2f}±{ot.std():.2f} classes {sorted(ct)}  class-mix shift: {set(ct) - set(cd) or "none"}')
json.dump(dict(ds=DS, noise_ratio=float(a[:,0].mean()), rho=float(a[:,1].mean()), dropped=float(a[:,2].mean()),
               spread=float(p90-p10), lag_mean=float(lags.mean()), lag_gt10=int((lags>10).sum()), n_units=len(lags),
               top5=top5, overlap=len(set(top5) & set(SENS)), frozen_mass=float(sum(zr[names.index(s)] for s in SENS)),
               top5_mass=float(zr[order[:5]].sum()), life_cv=float(ld.std()/ld.mean()), onset_frac_sd=float(od.std()),
               class_shift=sorted(set(ct) - set(cd))), open(os.path.join(HERE, f'dsx_why_{DS}.json'), 'w'))
log('done')
