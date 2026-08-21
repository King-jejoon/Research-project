# preprocess_ncmapss.py
import numpy as np

def preprocess_ncmapss(A_dev, A_test):

    # dev data
    eng_dev   = A_dev[:, 0].astype(int)
    state_dev = A_dev[:, 3].astype(int)

    normal_ranges = []
    start = 0
    for i in range(1, len(state_dev)):
        if (state_dev[i] != state_dev[i - 1]) or (eng_dev[i] != eng_dev[i - 1]):
            end = i - 1
            if state_dev[start] == 1:
                normal_ranges.append((start, end))
            start = i
    end = len(state_dev) - 1
    if state_dev[start] == 1:
        normal_ranges.append((start, end))

    print("\n=== Normal Range summary (A_dev index) ===")
    print(f"Total normal ranges : {len(normal_ranges)}")
    for (s, e) in normal_ranges:
        print(f"  Engine {eng_dev[s]} : [{s}, {e}] (len={e - s + 1})")

    abnormal_ranges = []
    start = 0
    for i in range(1, len(state_dev)):
        if (state_dev[i] != state_dev[i - 1]) or (eng_dev[i] != eng_dev[i - 1]):
            end = i - 1
            if state_dev[start] != 1:
                abnormal_ranges.append((start, end))
            start = i
    end = len(state_dev) - 1
    if state_dev[start] != 1:
        abnormal_ranges.append((start, end))

    print("\n=== Abnormal Range summary (A_dev index) ===")
    print(f"Total abnormal ranges : {len(abnormal_ranges)}")
    for (s, e) in abnormal_ranges:
        print(f"  Engine {eng_dev[s]} : [{s}, {e}] (len={e - s + 1})")

    # DEV 전체 범위 (엔진별 전체 구간)
    dev_all_ranges = []
    start = 0
    for i in range(1, len(eng_dev)):
        if eng_dev[i] != eng_dev[i - 1]:
            dev_all_ranges.append((start, i - 1))
            start = i
    dev_all_ranges.append((start, len(eng_dev) - 1))

    print("\n=== DEV All Range summary (A_dev index) ===")
    print(f"Total dev ranges : {len(dev_all_ranges)}")
    for (s, e) in dev_all_ranges:
        print(f"  Engine {eng_dev[s]} : [{s}, {e}] (len={e - s + 1})")

    # test data
    eng_test   = A_test[:, 0].astype(int)
    state_test = A_test[:, 3].astype(int)

    test_abnormal_ranges = []
    start = 0
    for i in range(1, len(state_test)):
        if (state_test[i] != state_test[i - 1]) or (eng_test[i] != eng_test[i - 1]):
            end = i - 1
            if state_test[start] != 1:
                test_abnormal_ranges.append((start, end))
            start = i
    end = len(state_test) - 1
    if state_test[start] != 1:
        test_abnormal_ranges.append((start, end))

    print("\n=== Test Abnormal Range summary (A_test index) ===")
    print(f"Total test abnormal ranges : {len(test_abnormal_ranges)}")
    for (s, e) in test_abnormal_ranges:
        print(f"  Engine {eng_test[s]} : [{s}, {e}] (len={e - s + 1})")

    # TEST 전체 범위 (엔진별 전체 구간)
    test_all_ranges = []
    start = 0
    for i in range(1, len(eng_test)):
        if eng_test[i] != eng_test[i - 1]:
            test_all_ranges.append((start, i - 1))
            start = i
    test_all_ranges.append((start, len(eng_test) - 1))

    print("\n=== TEST All Range summary (A_test index) ===")
    print(f"Total test ranges : {len(test_all_ranges)}")
    for (s, e) in test_all_ranges:
        print(f"  Engine {eng_test[s]} : [{s}, {e}] (len={e - s + 1})")

    return {
        'normal_ranges':        normal_ranges,
        'abnormal_ranges':      abnormal_ranges,
        'test_abnormal_ranges': test_abnormal_ranges,
        'dev_all_ranges':       dev_all_ranges,
        'test_all_ranges':      test_all_ranges,
        'eng_dev':              eng_dev,
        'state_dev':            state_dev,
        'eng_test':             eng_test,
        'state_test':           state_test
    }