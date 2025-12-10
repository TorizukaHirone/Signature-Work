import os
import numpy as np
import matplotlib.pyplot as plt

# 从你之前的脚本中导入 run_simulation
# 确保 drone_sim_coop.py 和本文件在同一 codes 目录下
from drone_sim_coop import run_simulation

# ---------- 配置图像输出目录 ----------
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def run_density_experiments(density_list, n_steps=50, seeds_per_setting=5):
    """
    对不同无人机数量进行实验，比较 competitive vs cooperative 的表现。
    返回字典，包含两种机制的平均完成率和平均速度。
    """
    density_list = list(density_list)

    comp_completion = []
    coop_completion = []
    comp_mean_speed = []
    coop_mean_speed = []

    for N in density_list:
        comp_crs = []
        coop_crs = []
        comp_speeds = []
        coop_speeds = []

        for s in range(seeds_per_setting):
            # competitive
            ms_comp, cr_comp, _ = run_simulation(
                mode="competitive", n_drones=N, n_steps=n_steps, seed=1000 + s
            )
            # cooperative
            ms_coop, cr_coop, _ = run_simulation(
                mode="cooperative", n_drones=N, n_steps=n_steps, seed=2000 + s
            )

            comp_crs.append(cr_comp)
            coop_crs.append(cr_coop)
            comp_speeds.append(np.mean(ms_comp))
            coop_speeds.append(np.mean(ms_coop))

        comp_completion.append(np.mean(comp_crs))
        coop_completion.append(np.mean(coop_crs))
        comp_mean_speed.append(np.mean(comp_speeds))
        coop_mean_speed.append(np.mean(coop_speeds))

        print(f"N={N:2d} | Competitive CR={np.mean(comp_crs):.3f}, "
              f"Cooperative CR={np.mean(coop_crs):.3f}")

    results = {
        "densities": np.array(density_list),
        "comp_completion": np.array(comp_completion),
        "coop_completion": np.array(coop_completion),
        "comp_speed": np.array(comp_mean_speed),
        "coop_speed": np.array(coop_mean_speed),
    }
    return results


def plot_density_results(results):
    densities = results["densities"]
    comp_completion = results["comp_completion"]
    coop_completion = results["coop_completion"]
    comp_speed = results["comp_speed"]
    coop_speed = results["coop_speed"]

    # 1) 完成率 vs 密度
    plt.figure(figsize=(6, 4))
    plt.plot(densities, comp_completion, marker="o", label="Competitive")
    plt.plot(densities, coop_completion, marker="s", label="Cooperative")
    plt.xlabel("Number of drones")
    plt.ylabel("Completion rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.title("Completion Rate vs Density")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "completion_vs_density.png"), dpi=200)

    # 2) 平均速度 vs 密度
    plt.figure(figsize=(6, 4))
    plt.plot(densities, comp_speed, marker="o", label="Competitive")
    plt.plot(densities, coop_speed, marker="s", label="Cooperative")
    plt.xlabel("Number of drones")
    plt.ylabel("Average mean speed (m/s)")
    plt.grid(True)
    plt.legend()
    plt.title("Average Speed vs Density")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "speed_vs_density.png"), dpi=200)

    plt.show()


def main():
    # 这里可以调密度列表，例如 [5, 10, 20, 30]
    density_list = [5, 10, 20]

    results = run_density_experiments(
        density_list=density_list,
        n_steps=50,
        seeds_per_setting=10  # 每个密度跑 10 次求平均
    )
    plot_density_results(results)


if __name__ == "__main__":
    main()
