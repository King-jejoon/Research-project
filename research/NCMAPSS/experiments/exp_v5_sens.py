"""
exp_v5_sens.py — baseline-window sensitivity RE-SWEPT on the final
coverage-conditional gate curves (user decision 2026-08-10: every constant
documented on the conditional chain).

Design: the gate rule stays frozen exactly as deployed — per unit,
V = 19.0 if the 30 flight-hour trigger window holds < 10 cycles, else 20.2
(the trigger window is part of the frozen rule, NOT the swept variable).
Only the detector's baseline window is swept, clip fixed at 10 sigma,
identical conventions to the fixed-gate sweep (exp_v4_docfigs2.sens_errors).
Dev only, cached stats, no test contact.
Outputs: v5_sens_results.txt, v5_sens.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import meandrop
from exp_v4_gate_redesign import unit_data, curve

VD, VS, NB = 20.2, 19.0, 10
CYC = [3, 4, 5, 6, 8, 12, 20]
FH = [10, 15, 20, 25, 30, 50]
RES = os.path.join(HERE, 'v5_sens_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5 SENS — window sensitivity on the conditional-gate curves '
        '(rule frozen: nbase<10 -> V=19.0 else 20.2; clip 10 sigma)')
    D = unit_data()
    curves = [(curve(d, VS if d['nbase'] < NB else VD),
               d['dur'], d['ucyc'], d['true'], d['u']) for d in D]
    E, U = {}, {}
    for mode, wins in [('cyc', CYC), ('fh', FH)]:
        for wdw in wins:
            e, uu = [], []
            for cur, dur, ucyc, true_on, u in curves:
                if mode == 'cyc':
                    w = max(3, min(int(wdw), len(cur) - 5))
                else:
                    w = max(3, int(np.searchsorted(np.cumsum(dur),
                                                   float(wdw)) + 1))
                mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
                k = meandrop(np.maximum(cur, mu0 - 10 * sd0))
                det = (int(ucyc[k]) if (k is not None and k < len(ucyc))
                       else int(ucyc[-1] + 1))
                e.append(det - true_on); uu.append(u)
            E[(mode, wdw)] = np.array(e, float)
            U[(mode, wdw)] = np.array(uu, int)
    log('')
    log('pooled onset RMSE per window (27 dev cases):')
    for mode, wins in [('cyc', CYC), ('fh', FH)]:
        row = ['%s%s=%.2f' % (w, 'cy' if mode == 'cyc' else 'h',
                              float(np.sqrt((E[(mode, w)] ** 2).mean())))
               for w in wins]
        log('  ' + '  '.join(row))
    np.savez(os.path.join(HERE, 'v5_sens.npz'),
             **{f'{m}_{w}_e': E[(m, w)] for m, w in E},
             **{f'{m}_{w}_u': U[(m, w)] for m, w in U})
    log(f'\nwall={time.time()-t0:.1f} s')


if __name__ == '__main__':
    main()
