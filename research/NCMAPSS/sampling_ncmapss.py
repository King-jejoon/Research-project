# sampling_ncmapss.py
import numpy as np

def sample_ranges(target_engines, ranges, eng_arr, K=300):
    sampled_idx_list = []
    count_by_engine  = {e: 0 for e in target_engines}

    for eng_id in target_engines:
        for (s, e) in ranges:
            if eng_arr[s] != eng_id:
                continue
            idx = np.arange(s, e + 1)
            T   = len(idx)
            if T <= 0:
                continue
            k   = min(K, T)

            pos = np.linspace(T - 1, 0, num=k)
            pos = np.round(pos).astype(int)
            pos = np.sort(np.unique(pos))

            if pos.size < k:
                pool = np.setdiff1d(np.arange(T), pos)
                need = k - pos.size
                add  = np.round(np.linspace(pool.size - 1, 0, num=need)).astype(int)
                pos  = np.sort(np.concatenate([pos, pool[add]]))
            else:
                pos = np.sort(pos[:k])

            sampled_idx_list.append(idx[pos])
            count_by_engine[eng_id] += len(idx[pos])

    sampled_idx = np.sort(np.unique(np.concatenate(sampled_idx_list)))
    return sampled_idx, count_by_engine


def sample_from_ranges(A, abnormal_ranges, K):
    results = {
        'indices': [],
        'cycles': [],
        'range_info': []
    }

    for range_idx, (start, end) in enumerate(abnormal_ranges):
        range_data  = A[start:end+1]
        cycle_col   = range_data[:, 1].astype(int)
        unique_cycles = sorted(set(cycle_col))

        range_indices = []
        range_cycles  = []

        for cycle in unique_cycles:
            cycle_mask = cycle_col == cycle
            cycle_indices_in_range = [i for i, val in enumerate(cycle_mask) if val]

            n_samples = len(cycle_indices_in_range)
            if n_samples > 0:
                step = max(1, n_samples // K)
                sampled_local_indices  = cycle_indices_in_range[::step][:K]
                sampled_global_indices = [start + idx for idx in sampled_local_indices]

                range_indices.extend(sampled_global_indices)
                range_cycles.extend([cycle] * len(sampled_global_indices))

        results['indices'].extend(range_indices)
        results['cycles'].extend(range_cycles)

        range_info = {
            'range_index':    range_idx,
            'start':          start,
            'end':            end,
            'range_length':   end - start + 1,
            'unique_cycles':  unique_cycles,
            'n_cycles':       len(unique_cycles),
            'sampled_count':  len(range_indices),
            'sampled_indices': range_indices,
            'sampled_cycles': range_cycles
        }
        results['range_info'].append(range_info)

    return results


def run_sampling(A_dev, A_test, eng_dev, normal_ranges, abnormal_ranges, test_abnormal_ranges, K=20, N=10):

    # DEV normal 샘플링
    normal_engines = list(np.unique(A_dev[:, 0]).astype(int))
    dev_normal_idx, dev_count_by_engine = sample_ranges(normal_engines, normal_ranges, eng_dev, K)

    print("\n=== DEV sampling (normal state, from A_dev) ===")
    print(f"Target engines : {normal_engines}")
    print(f"Samples per range (K) : {K}")
    for eng_id in normal_engines:
        for (s, e) in normal_ranges:
            if eng_dev[s] != eng_id:
                continue
            T = e - s + 1
            print(f"  Engine {eng_id} | range [{s},{e}] | range_len={T} | sampled={min(K,T)}")
    print("\nDEV sampled count by engine:")
    for eng_id, cnt in dev_count_by_engine.items():
        print(f"  Engine {eng_id} : {cnt}")
    print(f"\nTotal DEV sampled indices: {len(dev_normal_idx)}")

    # DEV abnormal 샘플링
    dev_abnormal_idx = sample_from_ranges(A_dev, abnormal_ranges, N)
    print("\n=== DEV sampling (abnormal state, from A_dev) ===")
    print(f"Samples per cycle (N) : {N}")
    print(f"\nTotal DEV sampled indices: {len(dev_abnormal_idx['indices'])}")

    # TEST abnormal 샘플링
    test_abnormal_idx = sample_from_ranges(A_test, test_abnormal_ranges, N)
    print("\n=== TEST sampling (abnormal state, from A_test) ===")
    print(f"Samples per cycle (N) : {N}")
    print(f"\nTotal TEST sampled indices: {len(test_abnormal_idx['indices'])}")

    return {
        'dev_normal_idx':      dev_normal_idx,
        'dev_count_by_engine': dev_count_by_engine,
        'dev_abnormal_idx':    dev_abnormal_idx,
        'test_abnormal_idx':   test_abnormal_idx
    }