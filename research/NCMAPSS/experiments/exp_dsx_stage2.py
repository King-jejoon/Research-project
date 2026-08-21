"""
exp_dsx_stage2.py — DS01/DS02 transfer, PHASE 2: stage-2 healthy-range
chain on dev.  Dev only, no test contact.

Per seed: conditional-gate onsets from the frozen Phase-1b rule
({DS}v6_detsel.npz) on the cached stage-1 stats -> healthy window = all
cycles strictly before the onset -> 3,000 rows per unit stratified equally
over healthy cycles (rng 3000+sd, DS03 convention; pool smaller than the
budget -> take all, disclosed) -> MOGP refit (frozen internals) -> dev
point stats (chain grid, u*7+sd — exact pairing with the stage-1 stats).

Downstream: stage-2 ungated tables -> lambda grid RE-SWEEP on the stage-2
tables (DS03 workflow: the stage-1 recipe stays frozen by the tie-band
precedent unless it fails shape there; a differing composite top is logged
as a WARNING, no auto re-freeze) -> HI #2 chain with the stage-1 recipe
(med3, beta by the frozen dev-LOO NASA rule) -> dev LOO RUL.
SNR diagnostic (dev, ground-truth hs onsets, diagnostics only): floor /
end-of-life signal / SNR per unit-seed case, stage-1 vs stage-2 GP at the
same points — metric definitions byte-identical to exp_v5c_resid_quality.
Usage: DS=ds01|ds02 python3 exp_dsx_stage2.py
Outputs: {DS}v6_stage2_results.txt, {DS}v6_gp_hr_s{sd}.pt,
         {DS}v6_hr_stats_s{sd}.npz, {DS}v6_stage2.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_c3_detcov import fit as gp_fit, point_stats, SIDX
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, train_model_tail, \
    shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3
from exp_dsx_lib import load_stats, q75_curve, detect_p, gate_V, \
    tables_ungated, run_grid, point_stats_robust, point_stats_parallel

DS = os.environ.get('DS', 'ds01')
assert DS in ('ds01', 'ds02', 'ds07', 'ds08a', 'ds04', 'ds05', 'ds06', 'ds08c')
SEEDS = [int(x) for x in os.environ.get('SEEDS_ONLY', '0,1,2').split(',')]
NPER, NTOT_PER_UNIT = 200, 3000
NEND = 5
RES = os.path.join(HERE, f'{DS}v6_stage2_results' + (f'_s{os.environ["SEEDS_ONLY"]}' if 'SEEDS_ONLY' in os.environ else '') + '.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load_rule():
    Z = np.load(os.path.join(HERE, f'{DS}v6_detsel.npz'))
    return dict(NB=int(Z['NB'][0]), V_sparse=float(Z['V_sparse'][0]),
                V_dense=float(Z['V_dense'][0]), wval=float(Z['wval'][0]),
                wmode=str(Z['wmode'][0]), clip=float(Z['clip'][0]))


def cond_onsets(sd, rule):
    D = load_stats(DS, sd)
    out = {}
    for u, d in D.items():
        V = gate_V(d, rule)
        cur = q75_curve(d, V)
        out[u] = detect_p(cur, d['dur'], d['ucyc'], rule['wval'],
                          rule['wmode'], rule['clip'])
    return out


def build_hr_stats(sd, cache, ons):
    out_path = os.path.join(HERE, f'{DS}v6_hr_stats_s{sd}.npz')
    if os.path.exists(out_path):
        got = np.load(out_path)['onset_used']
        want = np.array([[u, ons[u]] for u in sorted(ons)], int)
        assert np.array_equal(got, want), \
            f'seed{sd}: cached hr stats from different onsets — delete'
        log(f'seed{sd}: hr stats cached (onsets match)')
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    Zc3 = {u: d for u, d in load_stats(DS, sd).items()}
    rng = np.random.default_rng(3000 + sd)
    tr = []
    for u in sorted(ons):
        o = ons[u]
        rows_u = np.where((unit == u) & (cyc < o))[0]
        cyc_u = cyc[rows_u]
        ucyc = np.unique(cyc_u)
        per = int(np.ceil(NTOT_PER_UNIT / len(ucyc)))
        got = []
        for c in ucyc:
            r = rows_u[cyc_u == c]
            got.append(rng.choice(r, min(per, len(r)), replace=False))
        got = np.concatenate(got)
        if len(got) > NTOT_PER_UNIT:
            got = rng.choice(got, NTOT_PER_UNIT, replace=False)
        log(f'seed{sd}: u{u} window cyc<{o} ({len(ucyc)} cycles, pool '
            f'{len(rows_u)}) -> {len(got)} sampled')
        tr.append(got)
    tr = np.concatenate(tr)
    log(f'seed{sd}: train total {len(tr)} rows')
    t0 = time.time()
    gp_path = os.path.join(HERE, f'{DS}v6_gp_hr_s{sd}.pt')
    gp = None
    if os.path.exists(gp_path):
        try:
            try:
                gp = torch.load(gp_path, weights_only=False)
            except TypeError:
                gp = torch.load(gp_path)
            log(f'seed{sd}: saved hr GP loaded (no refit)')
        except Exception as e:
            log(f'seed{sd}: saved GP load FAILED ({e}) — refitting')
    if gp is None:
        gp, nret = gp_fit(W[tr], X[tr])
        log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
        try:
            torch.save(gp, gp_path)
        except Exception as e:
            log(f'seed{sd}: model save FAILED ({e}) — continuing')
    out = {'train_idx': tr,
           'onset_used': np.array([[u, ons[u]] for u in sorted(ons)], int)}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        d3 = Zc3[int(u)]
        idx, cc = [], []
        for c in d3['ucyc']:
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r); cc.append(np.full(len(r), c))
        idx = np.concatenate(idx); cc = np.concatenate(cc)
        pred, dcv, ll = point_stats_parallel(gp, W[idx], X[idx], gp_path=gp_path, log=log)
        out[f'u{u}_cc'] = cc.astype(np.int32)
        out[f'u{u}_resid'] = (X[idx] - pred).astype(np.float32)
        out[f'u{u}_dc'] = dcv.astype(np.float32)
        out[f'u{u}_ll'] = ll.astype(np.float32)
        out[f'u{u}_ucyc'] = d3['ucyc'].astype(np.int32)
        out[f'u{u}_hours'] = d3['dur'].astype(np.float32)
        out[f'u{u}_onset'] = np.array([d3['onset']])
    np.savez(out_path, **out)
    log(f'seed{sd}: hr stats written ({time.time()-t0:.0f}s total)')


def snr_diag():
    """dev residual-quality diagnostic, metrics = exp_v5c_resid_quality."""
    fl_o, fl_c, sg_o, sg_c, ids = [], [], [], [], []
    for sd in SEEDS:
        Do = load_stats(DS, sd, 'c3')
        Dc = load_stats(DS, sd, 'hr')
        units = sorted(Do)
        pool = []
        for u in units:
            assert np.array_equal(Do[u]['cc'], Dc[u]['cc']), \
                f'seed{sd} u{u}: subsample mismatch — pairing broken'
            on = Do[u]['onset']
            uc = np.unique(Do[u]['cc'])
            To = np.stack([trim25(Do[u]['resid'][Do[u]['cc'] == c])
                           for c in uc])
            pool.append(To[uc < on])
        sig_ref = np.concatenate(pool).std(0) + 1e-12
        for u in units:
            on = Do[u]['onset']
            uc = np.unique(Do[u]['cc'])
            To = np.stack([trim25(Do[u]['resid'][Do[u]['cc'] == c])
                           for c in uc]) / sig_ref
            Tc = np.stack([trim25(Dc[u]['resid'][Dc[u]['cc'] == c])
                           for c in uc]) / sig_ref
            h = uc < on
            fl_o.append(float(np.sqrt((To[h] ** 2).mean())))
            fl_c.append(float(np.sqrt((Tc[h] ** 2).mean())))
            sg_o.append(float(np.abs(To[-NEND:]).mean()))
            sg_c.append(float(np.abs(Tc[-NEND:]).mean()))
            ids.append((sd, u))
    fl_o, fl_c = np.array(fl_o), np.array(fl_c)
    sg_o, sg_c = np.array(sg_o), np.array(sg_c)
    n = len(fl_o)
    log(f'\nSNR diagnostic, dev ({n} unit-seed cases, ground-truth onsets, '
        'diagnostics only)')
    log(f'  healthy floor RMS : c3 {fl_o.mean():.3f}  hr {fl_c.mean():.3f}  '
        f'ratio {np.mean(fl_c / fl_o):.2f}  '
        f'p={st.wilcoxon(fl_o, fl_c).pvalue:.1e}  '
        f'hr lower in {int((fl_c < fl_o).sum())}/{n}')
    log(f'  end-of-life signal: c3 {sg_o.mean():.3f}  hr {sg_c.mean():.3f}  '
        f'ratio {np.mean(sg_c / sg_o):.2f}')
    r_o, r_c = sg_o / fl_o, sg_c / fl_c
    log(f'  SNR signal/floor  : c3 {r_o.mean():.2f}  hr {r_c.mean():.2f}  '
        f'p={st.wilcoxon(r_o, r_c).pvalue:.1e}  '
        f'hr better in {int((r_c > r_o).sum())}/{n}')
    per_u = {}
    for (sd, u), a, b in zip(ids, r_o, r_c):
        per_u.setdefault(u, []).append(b / a)
    log('  per-unit SNR gain (hr/c3, mean over seeds): '
        + '  '.join(f'u{u} {np.mean(v):.2f}x'
                    for u, v in sorted(per_u.items())))


def main():
    open(RES, 'w').close()
    t00 = time.time()
    rule = load_rule()
    Zh = np.load(os.path.join(HERE, f'{DS}v6_hi1.npz'))
    L1, L2 = float(Zh['l1'][0]), float(Zh['l2'][0])
    log(f'{DS.upper()} V6 STAGE2 — healthy-range chain, rule {rule}, '
        f'stage-1 recipe ({L1}, {L2})')
    cache = np.load(os.path.join(HERE, f'cache_{DS}.npz'))
    ons = {}
    for sd in SEEDS:
        ons[sd] = cond_onsets(sd, rule)
        hs = {u: load_stats(DS, sd)[u]['onset'] for u in ons[sd]}
        log(f'seed{sd}: conditional onsets {ons[sd]} (hs truth {hs})')
    for sd in SEEDS:
        build_hr_stats(sd, cache, ons[sd])
    if os.environ.get('STATS_ONLY'):
        log('STATS_ONLY — stage-2 stats built, downstream left to the merge run')
        return

    # lambda grid re-sweep on the stage-2 tables (tie-band precedent)
    log('')
    log('lambda grid RE-SWEEP on stage-2 tables (med3-free)')
    data = {}
    for sd in SEEDS:
        units, raw, Zn, hrs, mu, sg = tables_ungated(DS, sd, 'hr')
        data[sd] = (units, Zn, hrs)
    R2, sel2, comp2 = run_grid(data, 'stage-2 grid', log)
    frozen = (L1, L2)
    # tie-band rule, formalized from the DS03 precedent (gap 0.010 retained):
    # the stage-1 recipe is retained if it passes shape 3/3 on the stage-2
    # tables and the composite top leads it by <= TIE; otherwise the chain
    # re-freezes to the stage-2 composite top (dev-only decision, disclosed).
    TIE = 0.02
    if frozen in comp2:
        gap = comp2[sel2] - comp2[frozen]
        log(f'stage-1 recipe {frozen}: comp {comp2[frozen]:.3f} '
            f'(shape {R2[frozen]["shape"]}/3), stage-2 top {sel2} comp '
            f'{comp2[sel2]:.3f}, gap {gap:.3f} (tie band {TIE})')
        if gap <= TIE:
            recipe = frozen
            log(f'RECIPE RETAINED: {frozen} (inside the tie band)')
        else:
            recipe = sel2
            log(f'RECIPE RE-FROZEN: {sel2} (stage-1 recipe outside the '
                'tie band on the stage-2 tables)')
    else:
        recipe = sel2
        log(f'RECIPE RE-FROZEN: {sel2} (stage-1 recipe {frozen} fails '
            'shape 3/3 on the stage-2 tables)')
    L1, L2 = recipe

    # HI #2 chain with the final stage-2 recipe
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    rms, nss, oks, bs, Ps, Ts, Fs = [], [], [], [], [], [], []
    for sd in SEEDS:
        units, Zn, hrs = data[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok)); bs.append(b)
        Ps.append(P); Ts.append(T); Fs.append(F)
        log(f'chain seed{sd}: dev LOO={rms[-1]:.2f} '
            f'shape={"OK" if ok else "FAIL"} beta={b}')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log('')
    log(f'HI #2 chain (healthy range, recipe ({L1}, {L2}), med3): dev LOO '
        f'= {np.mean(rms):.2f} ± {np.std(rms):.2f}  '
        f'NASA {np.mean(nss):.1f} ± {np.std(nss):.1f}  '
        f'shape {int(np.sum(oks))}/{len(SEEDS)}  beta={bs}')
    log('dev per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    log(f'reference: HI #1 chain dev {float(Zh["dev_rm"].mean()):.2f} ± '
        f'{float(Zh["dev_rm"].std()):.2f}')

    snr_diag()
    np.savez(os.path.join(HERE, f'{DS}v6_stage2.npz'), P=P, T=T, F=F,
             rms=np.array(rms), nss=np.array(nss), bs=np.array(bs),
             l1=np.array([L1]), l2=np.array([L2]),
             onsets=np.array([[sd, u, ons[sd][u]] for sd in SEEDS
                              for u in sorted(ons[sd])], int))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
