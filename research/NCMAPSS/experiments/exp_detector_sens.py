"""
exp_detector_sens.py — DETECTOR SENSITIVITY (paper appendix / final section).

Splits the old heterogeneous Table 6 (which mixed one cycle window with three
flight-hour windows on a single axis) into TWO clean sweeps, each with three
same-unit window sizes:

  Figure A  cycle-count windows      W in {8, 12, 20} cycles
  Figure B  flight-hour windows      H in {30, 50, 80} flight-hours

both crossed with clip strength c in {10, 12, 14, 16} sigma.

Score: onset-detection RMSE on the 9 development units x 3 seeds = 27 cases
per configuration (truth = hs-onset cycle).  Reuses the frozen detector
(gated q75-LL curve -> baseline window -> clip mu0 - c*sd0 -> mean-drop),
identical to exp_rul_r5.detect() except the window definition and clip factor
are swept.  No GP work: reads the cached rul_input_s{seed}.npz per-cycle
q75-LL curves directly, so the whole sweep is seconds.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = [0, 1, 2]
CYC_WINS = [8, 12, 20]           # cycle-count baseline windows  (Figure A)
FH_WINS = [30, 50, 80]           # flight-hour baseline windows   (Figure B)
CLIPS = [10, 12, 14, 16]         # clip strength (x sigma)
RES = os.path.join(HERE, 'detector_sens_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load_dev(sd):
    z = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k.split('_')[1]) for k in z.files
                    if k.startswith('dev_') and k.endswith('_ucyc')})
    return {u: dict(llq=z[f'dev_{u}_llq75'],
                    dur=z[f'dev_{u}_hours'],
                    ucyc=z[f'dev_{u}_ucyc'],
                    onset=int(z[f'dev_{u}_onset'][0])) for u in units}


def meandrop(x, kmin=3):
    n = len(x); s = np.std(x, ddof=1) + 1e-12; cs = np.cumsum(x)
    best = (None, -np.inf)
    for k in range(kmin, n - 4):
        m1 = cs[k - 1] / k; m2 = (cs[-1] - cs[k - 1]) / (n - k)
        t = np.sqrt(k * (n - k) / n) * (m1 - m2) / s
        if t > best[1]:
            best = (k, t)
    return best[0]


def detect(d, window, clip, mode):
    """window in cycles (mode='cyc') or flight-hours (mode='fh'); clip = c*sigma."""
    cur = d['llq']; dur = d['dur']
    if mode == 'cyc':
        w = max(5, min(int(window), len(cur) - 5))
    else:
        w = max(5, int(np.searchsorted(np.cumsum(dur), float(window)) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - clip * sd0)
    k = meandrop(x)
    return int(d['ucyc'][k]) if (k is not None and k < len(d['ucyc'])) else int(d['ucyc'][-1] + 1)


def sweep(mode, windows):
    """returns dict[(window,clip)] = (rmse, mean_delay) over 9 dev x 3 seeds."""
    out = {}
    for window in windows:
        for clip in CLIPS:
            errs = []
            for sd in SEEDS:
                dev = load_dev(sd)
                for u, d in dev.items():
                    det = detect(d, window, clip, mode)
                    errs.append(det - d['onset'])
            errs = np.array(errs, float)
            out[(window, clip)] = (float(np.sqrt((errs ** 2).mean())),
                                   float(errs.mean()))
    return out


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('DETECTOR SENSITIVITY  (onset RMSE, cycles; 9 dev units x 3 seeds = 27 cases/config)')
    log(f'  cycle windows {CYC_WINS} | flight-hour windows {FH_WINS} | clip {CLIPS} sigma')

    for mode, windows, tag in [('cyc', CYC_WINS, 'A) CYCLE-COUNT windows'),
                               ('fh', FH_WINS, 'B) FLIGHT-HOUR windows')]:
        res = sweep(mode, windows)
        log('')
        log(f'=== {tag} ===')
        unit = 'cyc' if mode == 'cyc' else 'flt-h'
        header = f'{"clip":>6} | ' + ' | '.join(f'{w:>5}{unit}' for w in windows)
        log(header); log('-' * len(header))
        for clip in CLIPS:
            row = f'{clip:>4} s | ' + ' | '.join(
                f'{res[(w, clip)][0]:>8.2f}' for w in windows)
            log(row)
        # best per mode
        bw, bc = min(res, key=lambda k: res[k][0])
        log(f'  best: window={bw}{unit} clip={bc}s  onset RMSE={res[(bw, bc)][0]:.2f}  '
            f'delay={res[(bw, bc)][1]:+.2f}')
        np.savez(os.path.join(HERE, f'detector_sens_{mode}.npz'),
                 windows=np.array(windows), clips=np.array(CLIPS),
                 rmse=np.array([[res[(w, c)][0] for c in CLIPS] for w in windows]),
                 delay=np.array([[res[(w, c)][1] for c in CLIPS] for w in windows]))
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
