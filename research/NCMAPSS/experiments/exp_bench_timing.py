"""exp_bench_timing.py — clean running-time measurement for the HI benchmark.

Run on an IDLE machine.  Reports, for every benchmarked HI model:

  training  fit the HI model on the 9 dev units (663 flights x 5 features)
  testing   HI curves of the 6 test units + first-passage RUL prediction

Protocol: 3 warm-up calls, then 9 timed repeats; median and inter-quartile
range are reported so a stray scheduling hiccup cannot move the number.
Machine state (thread limits, load average) is logged next to the results.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_bench_hi as B
from exp_rul_r23 import eval_units, FRACS

SEEDS = [0, 1, 2]
WARMUP, REPS = 3, 9
RES = os.path.join(HERE, 'bench_timing_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def timed(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.sort(ts)
    return float(np.median(ts)), float(ts[-3] - ts[2])       # median, IQR-ish


def main():
    sel = json.loads(np.load(os.path.join(HERE, 'bench_dev_sel.npz'),
                             allow_pickle=True)['sel'][0])
    open(RES, 'w').close()
    la = os.getloadavg()
    log('CLEAN TIMING RUN of the HI benchmark')
    log(f'  load average at start: {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}')
    log(f'  OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")}  '
        f'NTHREADS={os.environ.get("NTHREADS")}')
    log(f'  {WARMUP} warm-up + {REPS} timed repeats, median reported, '
        f'averaged over seeds {SEEDS}')
    log('')

    rows = {}
    for name, s in sel.items():
        hp, betas = s['hp'], s['beta']
        f_med, f_iqr, t_med, t_iqr, p_med, r_med = [], [], [], [], [], []
        for i, sd in enumerate(SEEDS):
            du, dR, dcov = B.load_split(sd, 'dev')
            tu, tR, tcov = B.load_split(sd, 'test')
            m = B.build(name, **hp)
            hi_dev = m.fit_all(dR, dcov, du, seed=sd)
            hi_te = {u: m.predict(tR[u], tcov[u]) for u in tu}
            ctr = {u: np.arange(len(hi_dev[u])) / 500.0 for u in du}
            cte = {u: np.arange(len(hi_te[u])) / 500.0 for u in tu}

            a, b = timed(lambda: m.fit(dR, dcov, du, sd))
            f_med.append(a); f_iqr.append(b)
            c, _ = timed(lambda: [m.predict(tR[u], tcov[u]) for u in tu])
            d, _ = timed(lambda: eval_units(betas[i], ctr, hi_dev, cte, hi_te,
                                            FRACS, None))
            p_med.append(c); r_med.append(d)
            t_med.append(c + d); t_iqr.append(0.0)
        rows[name] = dict(fit=float(np.mean(f_med)), fit_iqr=float(np.mean(f_iqr)),
                          pred=float(np.mean(p_med)), rul=float(np.mean(r_med)),
                          test=float(np.mean(t_med)))
        log(f'  {name:>8} measured')

    log('')
    log(f'{"model":>8} | {"TRAINING":>12} | {"HI pred":>10} {"RUL":>10} '
        f'{"TESTING":>11}')
    log('-' * 62)
    for name, r in rows.items():
        log(f'{name:>8} | {r["fit"]*1e3:10.2f}ms | {r["pred"]*1e3:8.2f}ms '
            f'{r["rul"]*1e3:8.2f}ms {r["test"]*1e3:9.2f}ms')
    log('')
    log(f'{"model":>8} | {"TRAINING (s)":>13} | {"TESTING (s)":>12}')
    log('-' * 40)
    for name, r in rows.items():
        log(f'{name:>8} | {r["fit"]:13.4f} | {r["test"]:12.4f}')
    la = os.getloadavg()
    log(f'\nload average at end: {la[0]:.2f} {la[1]:.2f} {la[2]:.2f}')
    with open(os.path.join(HERE, 'bench_timing.json'), 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
