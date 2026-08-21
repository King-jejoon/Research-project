# inspect_data_structure.py
# Inspect the N-CMAPSS data structure used by the sampling strategy,
# so it can be documented in the paper (Methodology - Sampling Strategy).
import h5py
import numpy as np

FILENAME = '/Users/a1/Desktop/project/pytorch_practice/data/N-CMAPSS_DS03-012.h5'


def u(dset):
    """Decode an HDF5 varname dataset into a clean python list of strings."""
    raw = np.array(dset)                      # materialise first
    flat = np.array(raw, dtype='U20').ravel() # then cast to unicode
    return [s.strip() for s in flat.tolist()]


with h5py.File(FILENAME, 'r') as hdf:
    # ---- 1. raw HDF5 dataset shapes / dtypes (no full load) ----
    print("=" * 64)
    print("1) HDF5 datasets (shape, dtype)")
    print("=" * 64)
    groups = ['W', 'X_s', 'X_v', 'T', 'Y', 'A']
    for split in ['dev', 'test']:
        for g in groups:
            key = f'{g}_{split}'
            d = hdf.get(key)
            if d is not None:
                print(f"  {key:10s} shape={d.shape}  dtype={d.dtype}")

    # ---- 2. variable names per array ----
    print("\n" + "=" * 64)
    print("2) Variable names per array")
    print("=" * 64)
    W_var   = u(hdf.get('W_var'))
    X_s_var = u(hdf.get('X_s_var'))
    X_v_var = u(hdf.get('X_v_var'))
    T_var   = u(hdf.get('T_var'))
    A_var   = u(hdf.get('A_var'))
    print(f"  W_var   ({len(W_var)})  : {W_var}")
    print(f"  X_s_var ({len(X_s_var)}): {X_s_var}")
    print(f"  X_v_var ({len(X_v_var)}): {X_v_var}")
    print(f"  T_var   ({len(T_var)}) : {T_var}")
    print(f"  A_var   ({len(A_var)}) : {A_var}")

    # ---- 3. auxiliary matrix A: column meaning + engine / state breakdown ----
    print("\n" + "=" * 64)
    print("3) Auxiliary matrix A (used by the sampling strategy)")
    print("=" * 64)
    A_dev  = np.array(hdf.get('A_dev'))
    A_test = np.array(hdf.get('A_test'))
    print(f"  A_dev  shape={A_dev.shape}  columns={A_var}")
    print(f"  A_test shape={A_test.shape}")

    # column indices (per preprocess_ncmapss.py): 0 = unit, 1 = cycle, 3 = hs(state)
    for name, A in [('DEV', A_dev), ('TEST', A_test)]:
        unit  = A[:, 0].astype(int)
        cycle = A[:, 1].astype(int)
        state = A[:, 3].astype(int)
        units = np.unique(unit)
        print(f"\n  [{name}] rows={A.shape[0]:,}  units={list(units)}  "
              f"n_units={len(units)}")
        print(f"        total cycles (sum over units) = "
              f"{sum(len(np.unique(cycle[unit==uu])) for uu in units)}")
        print(f"        hs==1 (healthy) rows = {(state==1).sum():,}  "
              f"hs==0 (degraded) rows = {(state!=1).sum():,}")
        # per-unit cycle count
        for uu in units:
            m = unit == uu
            print(f"          unit {uu:2d}: rows={m.sum():>9,}  "
                  f"cycles={len(np.unique(cycle[m])):>3d}  "
                  f"healthy_rows={(state[m]==1).sum():>9,}")

    # ---- 4. within-cycle structure for ONE engine (paper table) ----
    print("\n" + "=" * 64)
    print("4) Within-cycle structure for a single engine (DEV unit 1)")
    print("=" * 64)
    UNIT = 1
    m = A_dev[:, 0].astype(int) == UNIT
    cy = A_dev[m, 1].astype(int)
    hs = A_dev[m, 3].astype(int)
    cycles = np.unique(cy)
    # (cycle, #time steps, hs) per cycle
    rec = [(int(c), int((cy == c).sum()), int(hs[cy == c][0])) for c in cycles]
    # locate the hs 1 -> 0 transition
    trans = next(i for i in range(1, len(rec)) if rec[i][2] != rec[i - 1][2])
    total_steps = sum(r[1] for r in rec)
    print(f"  unit {UNIT}: {len(rec)} cycles, {total_steps:,} time steps total")
    print(f"  {'cycle':>6} {'T_n,c':>8} {'hs':>3}")
    show = [0, 1, trans - 1, trans, len(rec) - 2, len(rec) - 1]
    for j, i in enumerate(show):
        if j in (2, 4):
            print(f"  {'...':>6} {'...':>8} {'..':>3}")
        c, t, h = rec[i]
        print(f"  {c:>6} {t:>8,} {h:>3}")
    print(f"  {'Total':>6} {total_steps:>8,}   (hs switches 1->0 at cycle {rec[trans][0]})")
