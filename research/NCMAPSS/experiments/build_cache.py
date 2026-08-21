"""
실험 가속용 캐시 빌더.
기존 모듈(load/preprocess/sampling)을 '호출'만 해서 필요한 배열·인덱스를 npz로 저장한다.
기존 코드는 수정하지 않는다.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import h5py
from preprocess_ncmapss import preprocess_ncmapss
from sampling_ncmapss   import run_sampling

FILENAME = os.path.join(__import__('exp_paths').DATA, 'N-CMAPSS_DS03-012.h5')
CACHE    = os.path.join(HERE, 'cache.npz')
K, N_ABN = 20, 10


def main():
    t0 = time.time()
    print('[cache] reading needed datasets (float32)...')
    with h5py.File(FILENAME, 'r') as h:
        W_dev    = np.array(h['W_dev'],    dtype=np.float32)
        X_s_dev  = np.array(h['X_s_dev'],  dtype=np.float32)
        A_dev    = np.array(h['A_dev'],    dtype=np.float32)
        W_test   = np.array(h['W_test'],   dtype=np.float32)
        X_s_test = np.array(h['X_s_test'], dtype=np.float32)
        A_test   = np.array(h['A_test'],   dtype=np.float32)
    print(f'[cache] read done {time.time()-t0:.1f}s  W_dev={W_dev.shape}')

    prep = preprocess_ncmapss(A_dev, A_test)
    normal_ranges  = prep['normal_ranges']
    dev_all_ranges = prep['dev_all_ranges']
    test_all_ranges= prep['test_all_ranges']
    eng_dev        = prep['eng_dev']

    # 노트북 셀4와 동일: abnormal 자리에 dev_all_ranges / test_all_ranges 전달
    samp = run_sampling(A_dev, A_test, eng_dev,
                        normal_ranges, dev_all_ranges, test_all_ranges,
                        K=K, N=N_ABN)

    dev_normal_idx    = np.asarray(samp['dev_normal_idx'], dtype=np.int64)
    dev_abn_indices   = np.asarray(samp['dev_abnormal_idx']['indices'], dtype=np.int64)
    dev_abn_cycles    = np.asarray(samp['dev_abnormal_idx']['cycles'],  dtype=np.int64)
    test_abn_indices  = np.asarray(samp['test_abnormal_idx']['indices'], dtype=np.int64)
    test_abn_cycles   = np.asarray(samp['test_abnormal_idx']['cycles'],  dtype=np.int64)

    def ranges_to_arr(r):
        return np.asarray(r, dtype=np.int64) if len(r) else np.zeros((0, 2), np.int64)

    print('[cache] saving npz...')
    np.savez_compressed(
        CACHE,
        W_dev=W_dev, X_s_dev=X_s_dev, A_dev=A_dev,
        W_test=W_test, X_s_test=X_s_test, A_test=A_test,
        normal_ranges=ranges_to_arr(normal_ranges),
        dev_all_ranges=ranges_to_arr(dev_all_ranges),
        test_all_ranges=ranges_to_arr(test_all_ranges),
        dev_normal_idx=dev_normal_idx,
        dev_abn_indices=dev_abn_indices, dev_abn_cycles=dev_abn_cycles,
        test_abn_indices=test_abn_indices, test_abn_cycles=test_abn_cycles,
    )
    sz = os.path.getsize(CACHE) / 1e6
    print(f'[cache] done -> {CACHE}  ({sz:.0f} MB)  total {time.time()-t0:.1f}s')
    print(f'[cache] dev_normal_idx={len(dev_normal_idx)}  dev_abn={len(dev_abn_indices)}  test_abn={len(test_abn_indices)}')


if __name__ == '__main__':
    main()
