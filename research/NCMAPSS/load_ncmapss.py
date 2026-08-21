# load_ncmapss.py
import h5py
import numpy as np
import time

def load_ncmapss(filename):

    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev   = np.array(hdf.get('W_dev'))
        X_s_dev = np.array(hdf.get('X_s_dev'))
        X_v_dev = np.array(hdf.get('X_v_dev'))
        T_dev   = np.array(hdf.get('T_dev'))
        Y_dev   = np.array(hdf.get('Y_dev'))
        A_dev   = np.array(hdf.get('A_dev'))

        # Test set
        W_test   = np.array(hdf.get('W_test'))
        X_s_test = np.array(hdf.get('X_s_test'))
        X_v_test = np.array(hdf.get('X_v_test'))
        T_test   = np.array(hdf.get('T_test'))
        Y_test   = np.array(hdf.get('Y_test'))
        A_test   = np.array(hdf.get('A_test'))

        # Varnames
        W_var   = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var   = np.array(hdf.get('T_var'))
        A_var   = np.array(hdf.get('A_var'))

    # from np.array to list dtype U4/U5
    W_var   = list(np.array(W_var,   dtype='U20'))
    X_s_var = list(np.array(X_s_var, dtype='U20'))
    X_v_var = list(np.array(X_v_var, dtype='U20'))
    T_var   = list(np.array(T_var,   dtype='U20'))
    A_var   = list(np.array(A_var,   dtype='U20'))

    W   = np.concatenate((W_dev,   W_test),   axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T   = np.concatenate((T_dev,   T_test),   axis=0)
    Y   = np.concatenate((Y_dev,   Y_test),   axis=0)
    A   = np.concatenate((A_dev,   A_test),   axis=0)

    print("W shape: "   + str(W.shape))
    print("X_s shape: " + str(X_s.shape))
    print("X_v shape: " + str(X_v.shape))
    print("T shape: "   + str(T.shape))
    print("A shape: "   + str(A.shape))

    return {
        'W': W, 'X_s': X_s, 'X_v': X_v, 'T': T, 'Y': Y, 'A': A,
        'W_dev': W_dev, 'X_s_dev': X_s_dev, 'X_v_dev': X_v_dev,
        'T_dev': T_dev, 'Y_dev': Y_dev, 'A_dev': A_dev,
        'W_test': W_test, 'X_s_test': X_s_test, 'X_v_test': X_v_test,
        'T_test': T_test, 'Y_test': Y_test, 'A_test': A_test,
        'W_var': W_var, 'X_s_var': X_s_var, 'X_v_var': X_v_var,
        'T_var': T_var, 'A_var': A_var
    }