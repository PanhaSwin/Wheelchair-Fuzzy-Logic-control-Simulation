Fuzzy EMA Controller for Dynamic Obstacle Avoidance

This simulation models an ego agent navigating in a 2D plane while predicting and reacting to the motion of a dynamic obstacle. A fuzzy logic controller is used to assess risk based on predicted proximity and motion direction. The output is smoothed using Exponential Moving Average (EMA) to produce a stable response.

Features

Ego and Obstacle Dynamics:

Ego: Constant velocity

Obstacle: Curved arc movement

Noisy Position Measurements:

Gaussian noise simulates sensor uncertainty

Prediction Mechanism:

Relative positions are predicted up to a configurable future horizon

Both ego and obstacle are projected using kinematic equations

Fuzzy Inference System:

Inputs: Distance and angle between ego and obstacle

Outputs: Action strength recommendation (risk level)

EMA Tracking:

Fuzzy outputs are smoothed using EMA to reduce reactivity to noise

Visualization:

Ego and obstacle trajectories

Predicted paths

Relative distance trends

Fuzzy & EMA controller output evolution

Requirements

Python 3.8+

numpy

matplotlib

scikit-fuzzy

Install dependencies:

pip install numpy matplotlib scikit-fuzzy

File Structure

Simulation.py: Main simulation script

fuzzy_map.py: generates fuzzy surface map

How It Works

Initialization: Positions and velocities of the ego and obstacle are initialized.

Simulation Loop:

Positions are updated step-by-step.

Predicted relative positions are calculated.

Fuzzy logic estimates action strength based on proximity and trajectory alignment.

EMA is applied to smooth the control output.

Plotting:

Trajectories and predictions

Sensor vs true vs predicted distance

Fuzzy and EMA output

Tuning Parameters

sigma_pos: Sensor noise magnitude

predict_horizon: Look-ahead prediction window

alpha: EMA smoothing factor

Fuzzy rule shapes and ranges
