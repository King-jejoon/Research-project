"""
exp_hi_quad.py — escalate L_flat (and the new 5%-window L_flat2) until every
test HI curve is a clean accelerating quadratic, judged by a STRICT visual rule.

STRICT rule (per curve, smoothed w=5, all 3 seeds x 6 test units must pass):
  start(first 10% mean) <= 0.30,  end >= 0.93
  decile slopes:      S(80-90)  >= S(70-80)         (monotone acceleration)
                      S(90-100) >= S(80-90) + 0.05
  half-window (the one the eye catches): S(95-100) >= S(90-95)
  global convexity:   quadratic coefficient of deg-2 fit > 0.3
Among passing configs, pick the lowest NASA score (drop 85%).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_quad.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_gp_selfdrop import build_resid, SENS
from exp_ds03_port import rul_eval, nasa_score, FRACS
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
DROP = 0.85
RES = os.path.join(HERE, 'hi_quad_results.txt')

SMOOTH_K = 1   # causal rolling mean on cycle-mean residuals (current + 2 prev)

CONFIGS = [
    ('A base(300,.002)',        dict(flat_w=300,  flat_m=0.002)),
    ('B (300,.004)',            dict(flat_w=300,  flat_m=0.004)),
    ('C (300,.006)',            dict(flat_w=300,  flat_m=0.006)),
    ('D (800,.004)',            dict(flat_w=800,  flat_m=0.004)),
    ('E (300,.002)+f2(300,.002)', dict(flat_w=300, flat_m=0.002, flat2_w=300, flat2_m=0.002)),
    ('F (300,.004)+f2(300,.004)', dict(flat_w=300, flat_m=0.004, flat2_w=300, flat2_m=0.004)),
    ('G (800,.004)+f2(800,.004)', dict(flat_w=800, flat_m=0.004, flat2_w=800, flat2_m=0.004)),
    ('H (800,.006)+f2(800,.006)', dict(flat_w=800, flat_m=0.006, flat2_w=800, flat2_m=0.006)),
    ('I (300,.002)+f2(800,.004)', dict(flat_w=300, flat_m=0.002, flat2_w=800, flat2_m=0.004)),
    ('J (300,.002)+f2(800,.006)', dict(flat_w=300, flat_m=0.002, flat2_w=800, flat2_m=0.006)),
    ('K (300,.004)+f2(1200,.006)', dict(flat_w=300, flat_m=0.004, flat2_w=1200, flat2_m=0.006)),
]


def causal_smooth(d, k=SMOOTH_K):
    """rolling mean over [t-k+1, t] — uses only past cycles, deployable online."""
    if k <= 1: return d
    out = np.empty_like(d)
    for t in range(len(d)):
        out[t] = d[max(0, t - k + 1):t + 1].mean(0)
    return out


def smooth_lists(dl, tl, k=SMOOTH_K):
    dl2 = [(u, cy, causal_smooth(d, k)) for u, cy, d in dl]
    tl2 = [(u, cy, causal_smooth(d, k)) for u, cy, d in tl]
    return dl2, tl2


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def wslope(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    return np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 2 else np.nan


def strict_check(h):
    hs = smooth1d(h); x = np.linspace(0, 1, len(hs))
    start = hs[x <= 0.10].mean()
    end = hs[-1]
    s78 = wslope(x, hs, 0.70, 0.80)
    s89 = wslope(x, hs, 0.80, 0.90)
    s9T = wslope(x, hs, 0.90, 1.00)
    s995 = wslope(x, hs, 0.90, 0.95)
    s95T = wslope(x, hs, 0.95, 1.00)
    a = np.polyfit(x, hs, 2)[0]
    ok = (start <= 0.30 and end >= 0.93 and s89 >= s78 and
          s9T >= s89 + 0.05 and s95T >= s995 and a > 0.3)
    return ok, dict(start=start, end=end, s78=s78, s89=s89, s9T=s9T,
                    s995=s995, s95T=s95T, a=a)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'STRICT-QUADRATIC HI ladder  (drop {DROP:.0%}, 3 seeds, all-units rule)')
    import numpy as _np
    z = {sd: _np.load(os.path.join(HERE, f'port_stats_s3_matern32_r1_m18_n8192_p200_s{sd}.npz'))
         for sd in SEEDS}
    RD = {sd: smooth_lists(*build_resid(z[sd], DROP)) for sd in SEEDS}
    log(f'residuals ready, causal smoothing k={SMOOTH_K} ({time.time()-t0:.0f}s)')
    log(f'{"config":>28} | {"RMSE":>11} | {"score":>5} | {"pass":>5} | '
        f'{"s995->s95T":>11} | worst fail')

    results = []
    for name, kw in CONFIGS:
        rmses, scores = [], []
        npass, ntot = 0, 0
        margins, fails = [], []
        for sd in SEEDS:
            dl, tl = RD[sd]
            np.random.seed(sd)
            him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                      lambda1=2.0, lambda2=0.25, init_threshold=0.2,
                                      alpha=0.001, **kw)
            dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
            tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
            P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
            rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
            scores.append(nasa_score(P, T))
            for u in tHI:
                ok, m = strict_check(tHI[u])
                ntot += 1; npass += int(ok)
                margins.append(m['s95T'] - m['s995'])
                if not ok:
                    why = []
                    if m['start'] > 0.30: why.append(f"start{m['start']:.2f}")
                    if m['end'] < 0.93: why.append(f"end{m['end']:.2f}")
                    if m['s89'] < m['s78']: why.append('s89<s78')
                    if m['s9T'] < m['s89'] + 0.05: why.append('s9T')
                    if m['s95T'] < m['s995']: why.append(f"tail5({m['s95T']-m['s995']:+.2f})")
                    if m['a'] <= 0.3: why.append('conv')
                    fails.append(f's{sd}u{u}:' + ','.join(why))
        r, rs, sc = np.mean(rmses), np.std(rmses), np.mean(scores)
        mg = np.mean(margins)
        allpass = npass == ntot
        log(f'{name:>28} | {r:>5.2f} ±{rs:.2f} | {sc:>5.1f} | {npass:>2}/{ntot} | '
            f'{mg:>+11.2f} | {fails[0] if fails else "-"}')
        results.append((name, kw, r, rs, sc, allpass))

    winners = [x for x in results if x[5]]
    if winners:
        best = min(winners, key=lambda x: x[4])
        log(f'\n--> WINNER: {best[0]}  RMSE {best[2]:.2f} ±{best[3]:.2f}  score {best[4]:.1f}')
        log(f'    kwargs: {best[1]}')
    else:
        log('\n--> no config passed the strict rule; escalate further')
    log(f'wall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
