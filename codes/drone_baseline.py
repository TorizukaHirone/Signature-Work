import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- Parameters ----------------
N_DRONES = 10
N_STEPS = 50
DT = 1.0
TRACK_LENGTH = 100.0
N_LANES = 2
LANES = np.arange(N_LANES)
SEGMENTS = 5
SEG_LENGTH = TRACK_LENGTH / SEGMENTS

SPEED_OPTIONS = np.array([0.8, 1.0, 1.2])  # m/s
D_MIN = 2.0  # minimum separation (for penalty)
FINISH_X = TRACK_LENGTH

np.random.seed(0)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------- State ----------------
x = np.random.uniform(0, 10, size=N_DRONES)  # initial positions
lane = np.random.choice(LANES, size=N_DRONES)  # 0 or 1
v = np.ones(N_DRONES)  # init speed 1.0

mean_speeds = []
finished = np.zeros(N_DRONES, dtype=bool)

# occupancy counter: lanes x segments
occupancy_counts = np.zeros((N_LANES, SEGMENTS))

# ---------------- Simulation Loop ----------------
for t in range(N_STEPS):
    # record mean speed
    mean_speeds.append(np.mean(v))

    # occupancy
    for i in range(N_DRONES):
        if finished[i]:
            continue
        seg_idx = int(np.clip(x[i] / SEG_LENGTH, 0, SEGMENTS - 1))
        occupancy_counts[lane[i], seg_idx] += 1

    # action selection (greedy self-interest)
    new_v = v.copy()
    new_x = x.copy()

    for i in range(N_DRONES):
        if finished[i]:
            continue

        best_score = -1e9
        best_speed = v[i]

        for cand_speed in SPEED_OPTIONS:
            cand_x = x[i] + cand_speed * DT

            # progress score
            progress = cand_speed

            # penalty if too close to a drone ahead in same lane
            penalty = 0.0
            for j in range(N_DRONES):
                if i == j or finished[j]:
                    continue
                if lane[j] != lane[i]:
                    continue
                # consider only drones ahead
                if x[j] > x[i]:
                    dist = (x[j] - cand_x)
                    if 0 < dist < D_MIN:
                        penalty -= 5.0  # arbitrary big penalty

            score = progress + penalty
            if score > best_score:
                best_score = score
                best_speed = cand_speed

        new_v[i] = best_speed
        new_x[i] = x[i] + best_speed * DT

    x = new_x
    v = new_v

    # mark finished
    finished = finished | (x >= FINISH_X)

# ---------------- Metrics ----------------
mean_speeds = np.array(mean_speeds)
completion_rate = np.mean(finished)
avg_occupancy = occupancy_counts / N_STEPS  # average over time

print("Completion rate:", completion_rate)
print("Average occupancy matrix (lanes x segments):")
print(avg_occupancy)

# ---------------- Plots ----------------

# 1) Mean speed vs time
plt.figure(figsize=(6, 4))
plt.plot(mean_speeds, marker='o')
plt.xlabel("Time step")
plt.ylabel("Mean speed (m/s)")
plt.title("Mean Speed over Time (Competitive Baseline)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mean_speed.png"), dpi=200)

# 2) Completion rate bar
plt.figure(figsize=(4, 4))
plt.bar(["Competitive"], [completion_rate])
plt.ylim(0, 1)
plt.ylabel("Completion rate")
plt.title("Completion Rate (Competitive Baseline)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "completion_rate.png"), dpi=200)

# 3) Congestion heatmap
plt.figure(figsize=(6, 3))
plt.imshow(avg_occupancy, aspect='auto', origin='lower',
           extent=[0.5, SEGMENTS + 0.5, -0.5, N_LANES - 0.5])
plt.colorbar(label="Avg occupancy")
plt.xlabel("Segment index")
plt.ylabel("Lane index")
plt.yticks(LANES)
plt.title("Average Occupancy Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "occupancy_heatmap.png"), dpi=200)

plt.show()
