# Fundamental of Artificial Intelligence - Phase 1 Projects

This repository contains three distinct AI projects covering fundamental concepts in search algorithms, machine learning, and optimization techniques.

## Project Overview

### 1. **Angry Birds: Star Wars - Search Algorithms**
**Objective**: Help Luke Skywalker navigate through a grid-based environment to collect all eggs while avoiding obstacles and enemies.

**Key Components**:
- Grid-based environment with various obstacles (bushes, boxes, moving pigs)
- Multiple search algorithms to implement:
  - **BFS** (Breadth-First Search)
  - **UCS** (Uniform Cost Search) 
  - **DLS** (Depth-Limited Search)
  - **A*** (with custom heuristic)
- Environment interaction through provided helper functions
- Evaluation based on expanded nodes and path efficiency

**Technologies**: PyGame, NumPy

### 2. **Linear Regression - Asteroid Diameter Prediction**
**Objective**: Develop a linear regression model to predict asteroid diameters using orbital and physical characteristics.

**Key Components**:
- Dataset containing asteroid features (orbital parameters, absolute magnitude, albedo, etc.)
- Implementation of **Stochastic Gradient Descent (SGD)** from scratch
- Data preprocessing and exploratory data analysis
- Model evaluation using R², MAE, and MSE metrics
- Optional advanced features: momentum, learning rate scheduling, early stopping, regularization

### 3. **DNA Center Finding - Local Search Algorithms** 
**Objective**: Find the central DNA string that minimizes the maximum Hamming distance to all strings in a set using local search algorithms.

**Key Components**:
- Implementation of two local search algorithms:
  - **Hill Climbing** (greedy local search)
  - **Simulated Annealing** (probabilistic optimization)
- Problem formulation based on Closest String Problem
- Functions for neighbor generation and cost calculation
- Comparison with brute-force approach
- Convergence analysis and parameter tuning

## Learning Objectives

- **Search Algorithms**: Understand and implement uninformed and informed search strategies
- **Machine Learning**: Build regression models from scratch with optimization algorithms
- **Optimization**: Apply local search techniques to combinatorial problems
- **Problem Solving**: Develop heuristic functions and analyze algorithm performance


