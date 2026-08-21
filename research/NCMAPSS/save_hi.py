# save_hi.py
import numpy as np
import pandas as pd


def save_HI_to_csv(model, dev_normalized, test_normalized, selected_sensors,
                   dev_path='dev_HI.csv', test_path='test_HI.csv'):

    def build_records(normalized_dict):
        records = []
        units = sorted(normalized_dict.keys())
        for unit in units:
            unit_dict = normalized_dict[unit]
            cycles = sorted(next(iter(unit_dict.values())).keys())
            data = np.array([
                [unit_dict[name][cycle] for name in selected_sensors]
                for cycle in cycles
            ])
            HI = model.forward(data).flatten()
            for cycle, hi in zip(cycles, HI):
                records.append({'Unit': unit, 'HI': hi})
        return records

    # DEV
    dev_records = build_records(dev_normalized)
    dev_df = pd.DataFrame(dev_records)
    dev_df.to_csv(dev_path, index=False)
    print(f"DEV saved: {dev_path} ({len(dev_df)} rows)")

    # TEST
    test_records = build_records(test_normalized)
    test_df = pd.DataFrame(test_records)
    test_df.to_csv(test_path, index=False)
    print(f"TEST saved: {test_path} ({len(test_df)} rows)")

    return dev_df, test_df