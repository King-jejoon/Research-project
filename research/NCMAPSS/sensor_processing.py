# sensor_processing.py
import numpy as np


def select_and_normalize(dev_residuals_by_unit, test_residuals_by_unit, selected_sensors):

    # DEV 센서 선택
    dev_selected = {}
    for unit, name_dict in dev_residuals_by_unit.items():
        dev_selected[unit] = {}
        for name in selected_sensors:
            if name in name_dict:
                dev_selected[unit][name] = name_dict[name]

    # TEST 센서 선택
    test_selected = {}
    for unit, name_dict in test_residuals_by_unit.items():
        test_selected[unit] = {}
        for name in selected_sensors:
            if name in name_dict:
                test_selected[unit][name] = name_dict[name]

    # DEV 기준 mean, std 계산
    sensor_stats = {}
    for name in selected_sensors:
        all_values = []
        for unit, name_dict in dev_selected.items():
            if name in name_dict:
                all_values.extend(name_dict[name].values())
        mean = np.mean(all_values)
        std  = np.std(all_values)
        sensor_stats[name] = {'mean': mean, 'std': std}
        print(f"{name}: mean={mean:.4f}, std={std:.4f}")

    # DEV 표준화
    dev_normalized = {}
    for unit, name_dict in dev_selected.items():
        dev_normalized[unit] = {}
        for name, cycle_dict in name_dict.items():
            mean = sensor_stats[name]['mean']
            std  = sensor_stats[name]['std']
            dev_normalized[unit][name] = {}
            for cycle, mean_val in cycle_dict.items():
                dev_normalized[unit][name][cycle] = (mean_val - mean) / std

    # TEST 표준화 (DEV 통계값 사용)
    test_normalized = {}
    for unit, name_dict in test_selected.items():
        test_normalized[unit] = {}
        for name, cycle_dict in name_dict.items():
            mean = sensor_stats[name]['mean']
            std  = sensor_stats[name]['std']
            test_normalized[unit][name] = {}
            for cycle, mean_val in cycle_dict.items():
                test_normalized[unit][name][cycle] = (mean_val - mean) / std

    return {
        'dev_selected':   dev_selected,
        'test_selected':  test_selected,
        'sensor_stats':   sensor_stats,
        'dev_normalized': dev_normalized,
        'test_normalized': test_normalized,
    }