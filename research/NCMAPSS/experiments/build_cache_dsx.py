"""build_cache_dsx.py — raw-array cache for a transfer subset (same six
keys the v6 chain consumes: W/X_s/A for dev and test, float32).
Usage: DS=ds08a|ds07 python3 build_cache_dsx.py"""
import os, sys, time
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.environ['DS']
FILES = {'ds07': 'N-CMAPSS_DS07.h5', 'ds08a': 'N-CMAPSS_DS08a-009.h5',
         'ds04': 'N-CMAPSS_DS04.h5', 'ds05': 'N-CMAPSS_DS05.h5',
         'ds06': 'N-CMAPSS_DS06.h5', 'ds08c': 'N-CMAPSS_DS08c-008.h5'}
src = os.path.join(__import__('exp_paths').DATA, FILES[DS])
t0 = time.time()
with h5py.File(src, 'r') as h:
    arr = {k: np.array(h[k], dtype=np.float32) for k in
           ['W_dev', 'X_s_dev', 'A_dev', 'W_test', 'X_s_test', 'A_test']}
out = os.path.join(HERE, f'cache_{DS}.npz')
np.savez(out, **arr)
print(f'{DS}: {out} written, W_dev {arr["W_dev"].shape}, '
      f'W_test {arr["W_test"].shape}, {time.time()-t0:.0f}s', flush=True)
