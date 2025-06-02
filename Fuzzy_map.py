import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
distance = ctrl.Antecedent(np.linspace(0, 6, 100), 'distance')
angle = ctrl.Antecedent(np.linspace(0, 180, 100), 'angle')
act_strength = ctrl.Consequent(np.linspace(0, 1, 100), 'act_strength')

# Membership functions
distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 1.2])
distance['close'] = fuzz.trimf(distance.universe, [0.8, 2.0, 3.0])
distance['far'] = fuzz.trimf(distance.universe, [2.5, 6, 6])

angle['converging'] = fuzz.trimf(angle.universe, [0, 0, 45])
angle['perpendicular'] = fuzz.trimf(angle.universe, [30, 90, 150])
angle['diverging'] = fuzz.trimf(angle.universe, [135, 180, 180])

act_strength['low'] = fuzz.trimf(act_strength.universe, [0, 0, 0.4])
act_strength['medium'] = fuzz.trimf(act_strength.universe, [0.3, 0.5, 0.7])
act_strength['high'] = fuzz.trimf(act_strength.universe, [0.6, 1.0, 1.0])

# Define rules
rule1 = ctrl.Rule(distance['very_close'], act_strength['high'])
rule2 = ctrl.Rule(distance['close'] & angle['converging'], act_strength['high'])
rule3 = ctrl.Rule(distance['close'] & angle['perpendicular'], act_strength['medium'])
rule4 = ctrl.Rule(distance['close'] & angle['diverging'], act_strength['low'])
rule5 = ctrl.Rule(distance['far'], act_strength['low'])

# Create control system and simulation
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# Create mesh grid for distance and angle input space
distances = np.linspace(0, 6, 50)
angles = np.linspace(0, 180, 50)
X, Y = np.meshgrid(distances, angles)
Z = np.zeros_like(X)

# Evaluate fuzzy controller over the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        fuzzy_sim.input['distance'] = X[i, j]
        fuzzy_sim.input['angle'] = Y[i, j]
        fuzzy_sim.compute()
        Z[i, j] = fuzzy_sim.output['act_strength']

# Plotting the fuzzy output surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Relative Angle (deg)')
ax.set_zlabel('Action Strength')
ax.set_title('Fuzzy Controller Surface Response')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()
