import numpy as np
import matplotlib.pyplot as plt
import os

# ================== Global Parameters ==================
DT = 1.0
TRACK_LENGTH = 100.0
N_LANES = 2
LANES = np.arange(N_LANES)
SEGMENTS = 5
SEG_LENGTH = TRACK_LENGTH / SEGMENTS

SPEED_OPTIONS = np.array([0.8, 1.0, 1.2])  # candidate speeds (m/s)
D_MIN = 2.0       # minimum separation distance for penalty
FINISH_X = TRACK_LENGTH
CAPACITY_PER_LANE = 10   # max drones per lane (for cooperative allocation)

RNG = np.random.default_rng(0)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


# ================== Core Step Functions ==================

def step_competitive(x, v, lane, finished):
    """
    One simulation step for the competitive baseline:
    each drone greedily chooses speed to maximize own progress
    while avoiding getting too close to the drone ahead in the same lane.
    """
    N = len(x)
    new_x = x.copy()
    new_v = v.copy()

    for i in range(N):
        if finished[i]:
            continue

        best_score = -1e9
        best_speed = v[i]

        for cand_speed in SPEED_OPTIONS:
            cand_x = x[i] + cand_speed * DT
            # progress reward
            progress = cand_speed

            # penalty if too close to a drone ahead in same lane
            penalty = 0.0
            for j in range(N):
                if i == j or finished[j]:
                    continue
                if lane[j] != lane[i]:
                    continue
                if x[j] > x[i]:
                    dist = x[j] - cand_x
                    if 0 < dist < D_MIN:
                        penalty -= 5.0  # large penalty

            score = progress + penalty
            if score > best_score:
                best_score = score
                best_speed = cand_speed

        new_v[i] = best_speed
        new_x[i] = x[i] + best_speed * DT

    return new_x, new_v, lane  # lane unchanged


def step_cooperative(x, v, lane, finished):
    """
    One simulation step for the cooperative mechanism:
    - Each drone forms a preference over candidate lanes using a simple utility.
    - Borda-style global lane allocation with capacity constraints.
    - Then each drone chooses best speed in its assigned lane.
    """
    N = len(x)
    new_lane = lane.copy()

    # ---------- 1. Each drone computes lane preferences ----------
    # candidate lanes: current, +/-1 (within bounds)
    lane_scores = {}  # (i, l) -> score
    for i in range(N):
        if finished[i]:
            continue

        current_lane = lane[i]
        candidate_lanes = [current_lane]
        if current_lane - 1 >= 0:
            candidate_lanes.append(current_lane - 1)
        if current_lane + 1 < N_LANES:
            candidate_lanes.append(current_lane + 1)

        # evaluate score for each candidate lane
        scores = []
        for L in candidate_lanes:
            best_score_for_lane = -1e9
            # choose best speed in that lane
            for cand_speed in SPEED_OPTIONS:
                cand_x = x[i] + cand_speed * DT
                progress = cand_speed

                penalty = 0.0
                for j in range(N):
                    if i == j or finished[j]:
                        continue
                    if lane[j] != L:   # consider drones currently in that lane
                        continue
                    if x[j] > x[i]:
                        dist = x[j] - cand_x
                        if 0 < dist < D_MIN:
                            penalty -= 5.0
                # simple lane "congestion" proxy: count drones already in that lane
                congestion = np.sum((lane == L) & (~finished))
                penalty -= 0.2 * congestion

                score = progress + penalty
                if score > best_score_for_lane:
                    best_score_for_lane = score

            scores.append((L, best_score_for_lane))

        # sort lanes by score descending to build preference order
        scores.sort(key=lambda t: t[1], reverse=True)
        # store: lane_scores[(i, L)] = Borda points
        # Borda: highest rank = len-1, next = len-2, ...
        k = len(scores)
        for rank, (L, _) in enumerate(scores):
            borda = k - 1 - rank
            lane_scores[(i, L)] = borda

    # ---------- 2. Global lane allocation with Borda points ----------
    # Build candidate list: (borda_score, i, lane)
    candidates = []
    for (i, L), score in lane_scores.items():
        candidates.append((score, i, L))
    # sort by Borda descending
    candidates.sort(key=lambda t: t[0], reverse=True)

    assigned_lane = {i: None for i in range(N)}
    lane_count = {L: 0 for L in LANES}

    for score, i, L in candidates:
        if finished[i]:
            continue
        if assigned_lane[i] is not None:
            continue
        if lane_count[L] >= CAPACITY_PER_LANE:
            continue
        assigned_lane[i] = L
        lane_count[L] += 1

    # update new_lane choices
    for i in range(N):
        if finished[i]:
            continue
        if assigned_lane[i] is not None:
            new_lane[i] = assigned_lane[i]

    # ---------- 3. Given assigned lane, pick best speed (like competitive) ----------
    new_x = x.copy()
    new_v = v.copy()
    for i in range(N):
        if finished[i]:
            continue

        best_score = -1e9
        best_speed = v[i]

        for cand_speed in SPEED_OPTIONS:
            cand_x = x[i] + cand_speed * DT
            progress = cand_speed

            penalty = 0.0
            for j in range(N):
                if i == j or finished[j]:
                    continue
                if new_lane[j] != new_lane[i]:
                    continue
                if x[j] > x[i]:
                    dist = x[j] - cand_x
                    if 0 < dist < D_MIN:
                        penalty -= 5.0

            score = progress + penalty
            if score > best_score:
                best_score = score
                best_speed = cand_speed

        new_v[i] = best_speed
        new_x[i] = x[i] + best_speed * DT

    return new_x, new_v, new_lane


# ================== Simulation Wrapper ==================

def run_simulation(mode="competitive", n_drones=10, n_steps=50, seed=0):
    """
    Run a full simulation and return metrics + occupancy matrix.
    mode: "competitive" or "cooperative"
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n_drones)
    lane = rng.integers(0, N_LANES, size=n_drones)
    v = np.ones(n_drones)
    finished = np.zeros(n_drones, dtype=bool)

    mean_speeds = []
    occupancy_counts = np.zeros((N_LANES, SEGMENTS))

    for t in range(n_steps):
        mean_speeds.append(np.mean(v))

        # record occupancy
        for i in range(n_drones):
            if finished[i]:
                continue
            seg_idx = int(np.clip(x[i] / SEG_LENGTH, 0, SEGMENTS - 1))
            occupancy_counts[lane[i], seg_idx] += 1

        # one step
        if mode == "competitive":
            x, v, lane = step_competitive(x, v, lane, finished)
        elif mode == "cooperative":
            x, v, lane = step_cooperative(x, v, lane, finished)
        else:
            raise ValueError("Unknown mode: {}".format(mode))

        # mark finished
        finished |= (x >= FINISH_X)

    mean_speeds = np.array(mean_speeds)
    completion_rate = np.mean(finished)
    avg_occupancy = occupancy_counts / n_steps

    return mean_speeds, completion_rate, avg_occupancy


# ================== Quick Experiment & Plots ==================

def main():
    n_drones = 10
    n_steps = 50

    # run both mechanisms
    ms_comp, cr_comp, occ_comp = run_simulation("competitive", n_drones, n_steps, seed=0)
    ms_coop, cr_coop, occ_coop = run_simulation("cooperative", n_drones, n_steps, seed=1)

    print("Competitive completion rate:", cr_comp)
    print("Cooperative completion rate:", cr_coop)
    print("Competitive average occupancy:\n", occ_comp)
    print("Cooperative average occupancy:\n", occ_coop)

    # 1) mean speed curves
    plt.figure(figsize=(6, 4))
    plt.plot(ms_comp, label="Competitive", marker='o')
    plt.plot(ms_coop, label="Cooperative", marker='s')
    plt.xlabel("Time step")
    plt.ylabel("Mean speed (m/s)")
    plt.title("Mean Speed over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mean_speed_compare.png"), dpi=200)

    # 2) completion rate bars
    plt.figure(figsize=(4, 4))
    plt.bar(["Competitive", "Cooperative"], [cr_comp, cr_coop])
    plt.ylim(0, 1)
    plt.ylabel("Completion rate")
    plt.title("Completion Rate Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "completion_rate_compare.png"), dpi=200)

    # 3) occupancy heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    im0 = axes[0].imshow(occ_comp, aspect='auto', origin='lower',
                         extent=[0.5, SEGMENTS + 0.5, -0.5, N_LANES - 0.5])
    axes[0].set_title("Competitive Occupancy")
    axes[0].set_xlabel("Segment index")
    axes[0].set_ylabel("Lane index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(occ_coop, aspect='auto', origin='lower',
                         extent=[0.5, SEGMENTS + 0.5, -0.5, N_LANES - 0.5])
    axes[1].set_title("Cooperative Occupancy")
    axes[1].set_xlabel("Segment index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "occupancy_compare.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
