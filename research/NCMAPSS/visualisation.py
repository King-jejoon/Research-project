import matplotlib.pyplot as plt
import numpy as np

def plot_HI_curves(model, dev_normalized, test_normalized, selected_sensors,
                   loss_history=None, lambda1=None, lambda2=None, epochs=None):

    def build_engine_data_list(normalized_dict):
        engine_data_list = []
        units = sorted(normalized_dict.keys())
        for unit in units:
            unit_dict = normalized_dict[unit]
            cycles = sorted(next(iter(unit_dict.values())).keys())
            data = np.array([
                [unit_dict[name][cycle] for name in selected_sensors]
                for cycle in cycles
            ])
            engine_data_list.append((unit, cycles, data))
        return engine_data_list

    dev_list  = build_engine_data_list(dev_normalized)
    test_list = build_engine_data_list(test_normalized)

    # lambda 정보 문자열
    info = []
    if lambda1 is not None:
        info.append(f"λ1={lambda1}")
    if lambda2 is not None:
        info.append(f"λ2={lambda2}")
    if epochs is not None:
        info.append(f"epochs={epochs}")
    info_str = "  |  " + ",  ".join(info) if info else ""

    cols = 3

    # DEV HI 곡선
    n_dev = len(dev_list)
    n_rows_dev = (n_dev + cols - 1) // cols
    fig, axes = plt.subplots(n_rows_dev, cols, figsize=(7 * cols, 6 * n_rows_dev))
    axes = np.array(axes).reshape(n_rows_dev, cols)
    fig.suptitle(f"DEV - HI Prediction per Engine{info_str}", fontsize=15, fontweight='bold')

    for idx, (unit, cycles, data) in enumerate(dev_list):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        HI = model.forward(data).flatten()
        ax.plot(cycles, HI, marker='o', markersize=4, linewidth=2, color='blue')
        ax.set_title(f"Engine {unit}", fontsize=13)
        ax.set_xlabel("Cycle", fontsize=11)
        ax.set_ylabel("HI", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_facecolor('white')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

    # 빈 subplot 숨기기
    for idx in range(n_dev, n_rows_dev * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()

    # TEST HI 곡선
    n_test = len(test_list)
    n_rows_test = (n_test + cols - 1) // cols
    fig, axes = plt.subplots(n_rows_test, cols, figsize=(7 * cols, 6 * n_rows_test))
    axes = np.array(axes).reshape(n_rows_test, cols)
    fig.suptitle(f"TEST - HI Prediction per Engine{info_str}", fontsize=15, fontweight='bold')

    for idx, (unit, cycles, data) in enumerate(test_list):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        HI = model.forward(data).flatten()
        ax.plot(cycles, HI, marker='o', markersize=4, linewidth=2, color='red')
        ax.set_title(f"Engine {unit}", fontsize=13)
        ax.set_xlabel("Cycle", fontsize=11)
        ax.set_ylabel("HI", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_facecolor('white')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

    # 빈 subplot 숨기기
    for idx in range(n_test, n_rows_test * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()