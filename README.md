# Fuzzy EMA Controller for Dynamic Obstacle Avoidance

This simulation models an ego agent navigating in a 2D plane while predicting and reacting to the motion of a dynamic obstacle. A fuzzy logic controller is used to assess risk based on predicted proximity and motion direction. The output is smoothed using Exponential Moving Average (EMA) to produce a stable response.

---

## Features

### Ego and Obstacle Dynamics

- **Ego**: Constant velocity  
- **Obstacle**: Curved arc movement

### Noisy Position Measurements

- Gaussian noise simulates sensor uncertainty

### Prediction Mechanism

- Relative positions are predicted up to a configurable future horizon  
- Both ego and obstacle are projected using kinematic equations

### Fuzzy Inference System

- **Inputs**: Distance and angle between ego and obstacle  
- **Output**: Action strength recommendation (risk level)

### EMA Tracking

- Fuzzy outputs are smoothed using EMA to reduce reactivity to noise

### Visualization

- Ego and obstacle trajectories  
- Predicted paths  
- Relative distance trends  
- Fuzzy & EMA controller output evolution

---

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `scikit-fuzzy`

### Install Dependencies

```bash
pip install numpy matplotlib scikit-fuzzy
