# residual_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def compute_residuals_by_unit(dev_residuals, test_residuals, dev_abnormal_idx, test_abnormal_idx, A_dev, A_test):

    dev_residuals_by_unit  = {}
    test_residuals_by_unit = {}

    # DEV data processing
    for residual_dict in dev_residuals:
        name     = residual_dict['name']
        residual = residual_dict['residual']
        indices  = dev_abnormal_idx['indices']
        cycles   = dev_abnormal_idx['cycles']

        units = [int(A_dev[idx, 0]) for idx in indices]

        for unit in set(units):
            if unit not in dev_residuals_by_unit:
                dev_residuals_by_unit[unit] = {}
            if name not in dev_residuals_by_unit[unit]:
                dev_residuals_by_unit[unit][name] = {}

            unit_mask      = np.array([u == unit for u in units])
            unit_cycles    = [c for c, m in zip(cycles, unit_mask) if m]
            unit_residuals = residual[unit_mask]

            for cycle in set(unit_cycles):
                cycle_mask     = np.array([c == cycle for c in unit_cycles])
                cycle_residuals = unit_residuals[cycle_mask]
                mean_residual  = np.mean(cycle_residuals)
                dev_residuals_by_unit[unit][name][cycle] = mean_residual

    # TEST data processing
    for residual_dict in test_residuals:
        name     = residual_dict['name']
        residual = residual_dict['residual']
        indices  = test_abnormal_idx['indices']
        cycles   = test_abnormal_idx['cycles']

        units = [int(A_test[idx, 0]) for idx in indices]

        for unit in set(units):
            if unit not in test_residuals_by_unit:
                test_residuals_by_unit[unit] = {}
            if name not in test_residuals_by_unit[unit]:
                test_residuals_by_unit[unit][name] = {}

            unit_mask      = np.array([u == unit for u in units])
            unit_cycles    = [c for c, m in zip(cycles, unit_mask) if m]
            unit_residuals = residual[unit_mask]

            for cycle in set(unit_cycles):
                cycle_mask      = np.array([c == cycle for c in unit_cycles])
                cycle_residuals = unit_residuals[cycle_mask]
                mean_residual   = np.mean(cycle_residuals)
                test_residuals_by_unit[unit][name][cycle] = mean_residual

    dev_units         = sorted(dev_residuals_by_unit.keys())
    test_units        = sorted(test_residuals_by_unit.keys())
    output_names_list = list(set([
        name
        for unit_dict in dev_residuals_by_unit.values()
        for name in unit_dict.keys()
    ]))

    return {
        'dev_residuals_by_unit':  dev_residuals_by_unit,
        'test_residuals_by_unit': test_residuals_by_unit,
        'dev_units':              dev_units,
        'test_units':             test_units,
        'output_names_list':      output_names_list,
    }


def plot_residuals(dataset, unit, dev_residuals_by_unit, test_residuals_by_unit, output_widget):
    with output_widget:
        output_widget.clear_output(wait=True)

        residuals_dict = dev_residuals_by_unit if dataset == 'DEV' else test_residuals_by_unit

        if unit not in residuals_dict:
            print(f"No data for unit {unit}")
            return

        unit_data = residuals_dict[unit]
        n_sensors = len(unit_data)
        n_rows    = (n_sensors + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, name in enumerate(sorted(unit_data.keys())):
            row = idx // 2
            col = idx % 2

            cycles = sorted(unit_data[name].keys())
            means  = [unit_data[name][c] for c in cycles]

            axes[row, col].plot(cycles, means, marker='o', linestyle='-', linewidth=2, markersize=6, alpha=0.7)
            axes[row, col].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axes[row, col].set_title(f"{dataset} - Unit {unit} - {name}", fontsize=11, fontweight='bold')
            axes[row, col].set_xlabel("Cycle")
            axes[row, col].set_ylabel("Mean Residual")
            axes[row, col].grid(True, alpha=0.3)

        for idx in range(n_sensors, n_rows * 2):
            axes.flatten()[idx].set_visible(False)

        plt.tight_layout()
        plt.show()


def show_residual_widgets(dev_residuals_by_unit, test_residuals_by_unit, dev_units, test_units):

    dataset_dropdown = widgets.Dropdown(
        options=['DEV', 'TEST'],
        value='DEV',
        description='Select Dataset:',
        disabled=False,
    )
    unit_dropdown = widgets.Dropdown(
        options=dev_units,
        value=dev_units[0],
        description='Select Unit:',
        disabled=False,
    )
    output_widget = widgets.Output()

    def update_units(change):
        if dataset_dropdown.value == 'DEV':
            unit_dropdown.options = dev_units
            unit_dropdown.value   = dev_units[0]
        else:
            unit_dropdown.options = test_units
            unit_dropdown.value   = test_units[0]

    def on_change(change):
        plot_residuals(
            dataset_dropdown.value, unit_dropdown.value,
            dev_residuals_by_unit, test_residuals_by_unit, output_widget
        )

    dataset_dropdown.observe(update_units, names='value')
    dataset_dropdown.observe(on_change,    names='value')
    unit_dropdown.observe(on_change,       names='value')

    display(dataset_dropdown)
    display(unit_dropdown)
    display(output_widget)

    plot_residuals(
        dataset_dropdown.value, unit_dropdown.value,
        dev_residuals_by_unit, test_residuals_by_unit, output_widget
    )