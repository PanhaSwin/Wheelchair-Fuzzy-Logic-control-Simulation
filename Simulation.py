import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#===== CONFIGURATION =====
dt = 0.1
T = 10.0
predict_horizon = 2.0
predict_step = 0.1

sigma_pos = 0.3
rng = np.random.default_rng(seed=42)

# ADJUST EGO POS VEL AND ACC HERE
ego_pos = np.array([0.0, 0.0])
ego_vel = np.array([0.7, 0.05])
ego_acc = np.array([0.0, 0.0])

# ADJUST STARTING POS TURNING RATE TURNING SPEED AND SPEED HERE
obs_pos = np.array([4.4, 6.2])
obs_heading = np.radians(-105)
obs_turn_rate = np.radians(12)
obs_speed = 1.3
obs_vel = obs_speed * np.array([np.cos(obs_heading), np.sin(obs_heading)])
obs_prev_vel = obs_vel.copy()
obs_acc = np.array([0.0, 0.0])

#===== LOGGING SETUP =====
time_log = []
dx_true_log, dy_true_log = [], []
dx_sensor_log, dy_sensor_log = [], []
dx_predicted_time_map, dy_predicted_time_map = {}, {}
ego_traj, obs_traj = [], []
ego_preds_all, obs_preds_all = [], []

#===== SIMULATION LOOP =====
for t in np.arange(0, T, dt):
    time_log.append(t)
    ego_traj.append(ego_pos.copy())
    obs_traj.append(obs_pos.copy())

    dx_true = ego_pos[0] - obs_pos[0]
    dy_true = ego_pos[1] - obs_pos[1]
    dx_true_log.append(dx_true)
    dy_true_log.append(dy_true)

    ego_noised = ego_pos + rng.normal(0, sigma_pos, 2)
    obs_noised = obs_pos + rng.normal(0, sigma_pos, 2)
    dx_sensor_log.append(ego_noised[0] - obs_noised[0])
    dy_sensor_log.append(ego_noised[1] - obs_noised[1])

    obs_acc = (obs_vel - obs_prev_vel) / dt
    obs_prev_vel = obs_vel.copy()

    ego_pred_points, obs_pred_points = [], []
    
    for tau in np.arange(dt, predict_horizon + dt, predict_step):
        if t + tau > T:
            continue

        ego_future = ego_noised + ego_vel * tau + 0.5 * ego_acc * (tau ** 2)
        obs_future = obs_noised + obs_vel * tau + 0.5 * obs_acc * (tau ** 2)

        dx_pred = ego_future[0] - obs_future[0]
        dy_pred = ego_future[1] - obs_future[1]

        dx_predicted_time_map.setdefault(t + tau, []).append(dx_pred)
        dy_predicted_time_map.setdefault(t + tau, []).append(dy_pred)

        ego_pred_points.append(ego_future)
        obs_pred_points.append(obs_future)

    ego_preds_all.append((t, ego_pred_points))
    obs_preds_all.append((t, obs_pred_points))

    ego_pos += ego_vel * dt
    obs_heading += obs_turn_rate * dt
    obs_vel = obs_speed * np.array([np.cos(obs_heading), np.sin(obs_heading)])
    obs_pos += obs_vel * dt

#===== FUZZY CONTROLLER SETUP =====
distance = ctrl.Antecedent(np.linspace(0, 6, 100), 'distance')
angle = ctrl.Antecedent(np.linspace(0, 180, 100), 'angle')
act_strength = ctrl.Consequent(np.linspace(0, 1, 100), 'act_strength')

# fUZZIFY I/O
distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 1.2])
distance['close'] = fuzz.trimf(distance.universe, [0.8, 2.0, 3.0])
distance['far'] = fuzz.trimf(distance.universe, [2.5, 6, 6])

angle['converging'] = fuzz.trimf(angle.universe, [0, 0, 45])
angle['perpendicular'] = fuzz.trimf(angle.universe, [30, 90, 150])
angle['diverging'] = fuzz.trimf(angle.universe, [135, 180, 180])

act_strength['low'] = fuzz.trimf(act_strength.universe, [0, 0, 0.4])
act_strength['medium'] = fuzz.trimf(act_strength.universe, [0.3, 0.5, 0.7])
act_strength['high'] = fuzz.trimf(act_strength.universe, [0.6, 1.0, 1.0])

# FUZZT RULES
rules = [
    ctrl.Rule(distance['very_close'], act_strength['high']),
    ctrl.Rule(distance['close'] & angle['converging'], act_strength['high']),
    ctrl.Rule(distance['close'] & angle['perpendicular'], act_strength['medium']),
    ctrl.Rule(distance['close'] & angle['diverging'], act_strength['low']),
    ctrl.Rule(distance['far'], act_strength['low'])
]

fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

#===== FUZZY + EMA CONTROL LOOP =====
fuzzy_time_log, fuzzy_output_log = [], []
ema_output_log, ema_output_actual = [], []
ema_output = 0.0

for i, (t_now, ego_preds) in enumerate(ego_preds_all):
    _, obs_preds = obs_preds_all[i]
    if not ego_preds or not obs_preds:
        continue

    predicted_distance = min(np.linalg.norm(e - o) for e, o in zip(ego_preds, obs_preds))
    rel_pos = obs_pos - ego_pos
    rel_vel = obs_vel - ego_vel
    cos_theta = np.clip(np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) * np.linalg.norm(rel_vel)), -1.0, 1.0)
    relative_angle_deg = np.degrees(np.arccos(cos_theta))

    fuzzy_sim.input['distance'] = predicted_distance
    fuzzy_sim.input['angle'] = relative_angle_deg
    fuzzy_sim.compute()
    fuzzy_output = fuzzy_sim.output['act_strength']

    fuzzy_time_log.append(t_now)
    fuzzy_output_log.append(fuzzy_output)

    alpha = 0.15
    ema_output = alpha * fuzzy_output + (1 - alpha) * ema_output
    ema_output_log.append(ema_output)

    ema_output_actual.append(ema_output)  # Placeholder for more advanced tracking if needed

#===== PLOTTING =====
# Traj
plt.figure(figsize=(10, 8))
ego_traj, obs_traj = np.array(ego_traj), np.array(obs_traj)
plt.plot(ego_traj[:, 0], ego_traj[:, 1], label='Ego Trajectory', color='blue')
plt.plot(obs_traj[:, 0], obs_traj[:, 1], label='Obstacle Trajectory', color='orange')

for i in range(0, len(ego_preds_all), 8):
    _, ego_preds = ego_preds_all[i]
    _, obs_preds = obs_preds_all[i]
    ego_preds, obs_preds = np.array(ego_preds), np.array(obs_preds)
    plt.plot(ego_preds[:, 0], ego_preds[:, 1], 'b--', alpha=0.4)
    plt.plot(obs_preds[:, 0], obs_preds[:, 1], 'r--', alpha=0.4)
    plt.scatter(ego_preds[0, 0], ego_preds[0, 1], color='blue', s=20)
    plt.scatter(obs_preds[0, 0], obs_preds[0, 1], color='red', s=20)

plt.scatter(ego_traj[0, 0], ego_traj[0, 1], c='green', label='Ego Start', s=60)
plt.scatter(obs_traj[0, 0], obs_traj[0, 1], c='red', label='Obstacle Start', s=60)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Trajectories with KF Predictions')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# dist and control (time)
mag_true_log = [np.linalg.norm([dx, dy]) for dx, dy in zip(dx_true_log, dy_true_log)]
mag_sensor_log = [np.linalg.norm([dx, dy]) for dx, dy in zip(dx_sensor_log, dy_sensor_log)]
mag_kf_time = sorted(dx_predicted_time_map)
mag_kf_avg = [
    np.mean([np.linalg.norm([dx, dy]) for dx, dy in zip(dx_predicted_time_map[t], dy_predicted_time_map[t])])
    for t in mag_kf_time
]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(time_log, mag_true_log, label='Distance (True)', color='black')
ax1.plot(time_log, mag_sensor_log, label='Distance (Sensor)', color='gray', linestyle='--')
ax1.plot(mag_kf_time, mag_kf_avg, label='Distance (KF Pred)', color='green', linestyle=':')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Distance (m)')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(fuzzy_time_log, fuzzy_output_log, label='Fuzzy Output', color='purple', linestyle='-.')
ax2.plot(fuzzy_time_log, ema_output_log, label='EMA Output', color='red')
ax2.set_ylabel('Action Strength')
ax2.set_ylim(0, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Distance & Controller Outputs')
plt.tight_layout()
plt.show()
