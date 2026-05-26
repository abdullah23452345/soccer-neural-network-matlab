# Soccer Neural Network in MATLAB

This project demonstrates a custom artificial neural network built in MATLAB using backpropagation and stochastic gradient descent to classify soccer shots as goals or misses.

## Project Overview

The neural network is trained on randomly generated soccer shot data. Shot locations closer to the goal line have a higher probability of resulting in a goal.

The model learns decision regions that classify whether a shot is likely to result in a goal or miss based on field position.

## Features

- Random soccer shot dataset generation
- Multi-layer neural network architecture
- Backpropagation training
- Stochastic gradient descent optimization
- Cost function visualization
- Decision boundary visualization

## Neural Network Architecture

Input Layer:
- x-position on field
- y-distance toward goal

Hidden Layers:
- 4 neurons
- 4 neurons

Output Layer:
- Goal
- Miss

## Technologies Used

- MATLAB
- Artificial Neural Networks
- Backpropagation
- Stochastic Gradient Descent
- Data Visualization

## Visualizations

### Training Progress
Shows the cost function decreasing during training.

### Decision Regions
Displays the learned classification regions for predicting goals vs misses.

## Files

- `netbp_soccer.m` — Main neural network implementation
- `soccer_net_training_progress.png` — Cost function decay during training
- `soccer_decision_regions.png` — Learned decision boundaries
- `soccer_shots.png` — Random soccer shot dataset
- `Mathematics Behind Deep Learning.pptx` — Presentation explaining neural network concepts

## Author

Abdullah Ahmed
Mathematics–Data Science Student
Wichita State University
