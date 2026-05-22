# Autonomous Delivery Robot: Q-Learning in a Grid-World Environment
**Project Documentation**

## 1. Introduction
This project simulates an autonomous delivery robot operating within a stylized 10x10 urban grid environment. The core objective is to employ **Reinforcement Learning (RL)**—specifically **Tabular Q-Learning**—to teach the robot an optimal policy for navigating from a central depot to execute complex multi-package delivery routing, while avoiding obstacles (buildings and planters). 

Rather than relying on deterministic pathfinding algorithms (such as A* or Dijkstra's), this project demonstrates how an agent can iteratively learn spatial navigation and task completion through trial and error, guided purely by an engineered reward function. The project leverages a modular architecture separating configuration, environment, training, and execution logic.

## 2. Environment Architecture (`DeliveryBot-v0`)
The simulation environment is implemented as a custom **Gymnasium** environment (`delivery_env.py`).

### 2.1 Spatial Dynamics
*   **Grid:** 10x10 discrete grid.
*   **Obstacles:** 
    *   16 impassable building cells configured in a block layout.
    *   8 impassable planter (grass) cells distributed across the environment (minimum Manhattan distance of 3 between each).
*   **Locations of Interest:**
    *   **Depot:** Fixed at `(0, 0)`. The robot always begins its episode here or at a valid random start location.
    *   **Pickup Nodes:** 4 fixed locations `[(1,1), (1,8), (8,1), (8,8)]`.
    *   **Dropoff Nodes:** 4 fixed locations `[(0,5), (5,0), (5,9), (9,5)]`.

### 2.2 Multi-Package Delivery System
The environment supports a complex multi-package delivery task:
*   **Active Packages:** All 4 packages are active simultaneously.
*   **Capacity:** The robot can carry a maximum of 2 packages at any given time.
*   **Randomized Destinations:** At the start of each episode, the mapping between the 4 pickup nodes and the 4 dropoff nodes is randomly shuffled (24 possible permutations). This ensures the robot learns a generalized policy rather than memorizing a single set of routes.

### 2.3 State Space
The environment must provide a Fully Observable Markov Decision Process (MDP). The discrete state index encodes:
1.  **Robot X-coordinate** (0-9)
2.  **Robot Y-coordinate** (0-9)
3.  **Package Statuses** (3 states per package: `0=waiting`, `1=carried`, `2=delivered` -> $3^4 = 81$ combinations)
4.  **Dropoff Permutation Index** (0-23)

*Total Theoretical State Space Size:* `10 * 10 * 81 * 24 = 194,400` discrete states.

### 2.4 Action Space
The agent can select from a discrete action space of 6 potential actions:
*   `0`: Move Up
*   `1`: Move Down
*   `2`: Move Left
*   `3`: Move Right
*   `4`: Pick up Package (from a waiting package at the current cell, if capacity allows)
*   `5`: Drop off Package (for a carried package matching the current dropoff cell)

### 2.5 Reward Function
The reward structure enforces efficiency, safety, and task completion:
*   **Valid Step (`-1`):** Applied on standard movement to encourage the shortest path.
*   **Illegal Action (`-10`):** Applied for collision with buildings or planters, invalid pickups (wrong location or at capacity), or invalid dropoffs.
*   **Successful Pickup (`+20`):** Intermediate positive reinforcement for acquiring a package.
*   **Successful Dropoff (`+100`):** Reward for completing a specific delivery job. The episode terminates when all 4 packages are successfully delivered.

## 3. Q-Learning Algorithm
The learning logic is executed in `train.py`. We utilize a Tabular Q-learning algorithm, updating the agent's action-value estimations according to the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### 3.1 Hyperparameters (Defined in `config.py`)
*   **Learning Rate ($\alpha$):** `0.1` — Controls the extent to which newly acquired information overrides old information.
*   **Discount Factor ($\gamma$):** `0.99` — Represents the importance of future rewards. A value near 1.0 makes the agent heavily weight long-term success over immediate gratification.
*   **Exploration Strategy:** $\epsilon$-greedy approach.
    *   `EPSILON_START`: `1.0` (100% random exploration initially).
    *   `EPSILON_DECAY`: `0.9999` (Multiplicative decay per episode, slowed down to accommodate the massive state space).
    *   `EPSILON_MIN`: `0.01` (Minimum 1% chance to explore).
*   **Training Duration:** `50,000` episodes, capped at `400` steps per episode.

## 4. Evaluation and Metrics
During training, the system logs the agent's performance, producing rolling-average visualizations.
*   **`rewards.png`**: Illustrates the agent's cumulative reward per episode.
*   **`steps.png`**: Depicts the number of steps required to terminate an episode. Over time, this curve drops as the agent learns the most direct, valid paths.
*   **`accuracy.png`**: Shows the delivery success rate (percentage of episodes where the robot successfully delivers all 4 packages).

## 5. Visualizer
The environment features a comprehensive `pygame`-based rendering engine (`run.py`). It maps the internal matrix representation to visual assets (roads, sidewalks, crosswalks, buildings, planters, and an animated robot). 

Unique visual features for the multi-package system:
*   Waiting packages are rendered in distinct colors (Gold, Teal, Pink, Lime).
*   Dropoff markers (crosshairs) match the color of their respective packages and pulse dynamically when the package is being carried.
*   The robot dynamically displays miniature colored boxes on its chassis representing its currently carried cargo.

## 6. Setup & Execution Instructions

### Prerequisites
*   Python 3.10+
*   Dependencies: `gymnasium`, `numpy`, `matplotlib`, `pygame`

### Running the Project
The entry point for the project is `main.py`. The environment requires an active python virtual environment containing the necessary dependencies.

1.  **Standard Run (Train + Demo):**
    ```bash
    python main.py
    ```
    This script will train the model for 50,000 episodes, export the learning plots, save the policy to `delivery_qtable.npy`, and spawn a Pygame window showing simulated test runs.

2.  **Headless Training (No Pygame Demo):**
    ```bash
    python main.py --skip-demo
    ```

3.  **Run Demo from Saved Policy:**
    ```bash
    python main.py --demo-only
    ```

4.  **CLI Arguments:**
    *   `--episodes <int>`: Number of training episodes.
    *   `--demo-episodes <int>`: Number of test runs to render.
    *   `--demo-step-delay <int>`: Time step delay (ms) for the visualization.

## 7. Conclusion
This simulation successfully models how reinforcement learning can be practically applied to complex autonomous navigation and capacity-constrained logistics. The Q-learning agent effectively maps a massive state space (194,400 states), determining an optimal, generalized routing policy capable of addressing any permutation of randomized task variables while respecting physical and payload constraints.