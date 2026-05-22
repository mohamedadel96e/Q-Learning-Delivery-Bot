# Autonomous Delivery Robot Using Q-Learning

## 1. Project Idea

This project simulates an autonomous delivery robot moving through a small urban environment. The robot starts at a depot, navigates a 10x10 city grid, and must execute a complex logistics run: it has to pick up 4 distinct packages from different locations and deliver them to their respective destinations. To make things harder, the robot can only carry a maximum of 2 packages at a time.

The goal is not only to make the robot move, but to make it learn a useful delivery strategy through reinforcement learning. Instead of manually programming every route, we let the robot explore the environment, manage its inventory, make mistakes, receive rewards or penalties, and gradually discover a reliable multi-stop delivery strategy.

## 2. The Central Question

Before writing code, we should ask:

> If a delivery robot is placed inside a city block with multiple pending deliveries and a limited carrying capacity, what information does it need in order to make a good decision?

At first, the answer might seem simple:

- Where is the robot?
- Which packages are waiting to be picked up?
- Which packages is the robot currently carrying?
- Which packages have already been delivered?
- Which destination belongs to which package?

These questions naturally lead us to a reinforcement learning formulation.

The robot is an agent. The city grid is the environment. Every time the robot chooses an action, the environment responds with a new state and a reward. Over many episodes, the robot learns which actions usually lead to successful completion of the entire job.

## 3. Environment Design

The project has been organized into a modular architecture:
- `config.py`: Hyperparameters and environment settings.
- `delivery_env.py`: The custom Gymnasium environment.
- `train.py`: The Q-learning training loop.
- `run.py`: The evaluation and rendering script.
- `main.py`: The primary entry point.

The environment is registered as:

```text
DeliveryBot-v0
```

### 3.1 City Grid

The city is represented as a 10x10 grid.

Each cell can represent part of the urban environment:

- Open road or sidewalk
- A building cell that blocks movement
- A planter (grass) cell that blocks movement
- The robot depot
- A package pickup location
- A customer dropoff location

### 3.2 Buildings and Obstacles

Buildings and planters are impassable. If the robot tries to move into an obstacle, the move is rejected and the robot receives a penalty. Planters are spaced out to create specific bottlenecks.

This is important because a delivery robot cannot simply move in a straight line to the destination. It has to learn paths around blocked areas.

### 3.3 Multi-Package Delivery System

At the start of every episode, there are 4 packages waiting at fixed pickup locations:

```text
(1, 1), (1, 8), (8, 1), (8, 8)
```

There are also 4 fixed dropoff locations:

```text
(0, 5), (5, 0), (5, 9), (9, 5)
```

To force the robot to learn a general policy rather than just memorizing one specific set of routes, the environment randomly shuffles the mapping between packages and dropoff locations. There are 4! (24) possible permutations. In one episode, package #1 might go to `(0, 5)`, but in the next, it might go to `(9, 5)`.

The robot also has a strict constraint: `MAX_CARRY = 2`. It cannot simply pick up all 4 packages and then drop them off. It must plan intermediate deliveries to free up space.

## 4. State Space

The state tells the robot what situation it is currently in.

Because of the multi-package nature of the problem, the state needs to be incredibly detailed. 

The environment encodes a massive discrete state index:

1. **Robot X-position** (0-9)
2. **Robot Y-position** (0-9)
3. **Status of Package 0** (Waiting, Carried, Delivered)
4. **Status of Package 1** (Waiting, Carried, Delivered)
5. **Status of Package 2** (Waiting, Carried, Delivered)
6. **Status of Package 3** (Waiting, Carried, Delivered)
7. **Dropoff Permutation Index** (0-23)

This results in a state space of:
`10 * 10 * (3^4) * 24 = 194,400` total states.

### Why This Matters

This is a key reinforcement learning idea: the state must contain enough information for the agent to make a good decision.

If the robot didn't know the Permutation Index, it wouldn't know which dropoff corresponds to the package it is holding. If it didn't know the status of all 4 packages, it couldn't decide whether to navigate toward a pickup or a dropoff. By encoding all this into a single integer, the Q-table can map every possible combination of events to an optimal action.

## 5. Action Space

The robot has 6 possible actions:

| Action ID | Meaning          |
| --------: | ---------------- |
|         0 | Move up          |
|         1 | Move down        |
|         2 | Move left        |
|         3 | Move right       |
|         4 | Pick up package  |
|         5 | Drop off package |

In this project, pickup and dropoff are explicit actions. The environment logic handles the capacity constraints. If the robot tries to pick up a 3rd package, the action fails and returns a penalty. The robot must learn when its inventory is full and route to a dropoff.

## 6. Reward Design

Rewards are how we communicate the objective to the robot.

The reward system is:

| Event                                | Reward |
| ------------------------------------ | -----: |
| Valid movement step                  |     -1 |
| Hitting a building or invalid action |    -10 |
| Successful pickup                    |    +20 |
| Successful dropoff                   |   +100 |

### Why Give -1 Per Step?

The `-1` step penalty asks:
> Can you solve the deliveries using fewer moves?

This encourages the fastest route, prioritizing efficient logistics over meandering paths.

### Why Penalize Illegal Actions?

Illegal actions include:
- Trying to move into a building or planter
- Trying to pick up a package when capacity is full (2/2)
- Trying to drop off at the wrong destination

The `-10` penalty teaches the robot to manage its inventory properly.

## 7. What Is Q-Learning?

Q-learning is a reinforcement learning algorithm that learns the value of taking an action in a state.

The main idea is:
> For each state, estimate how good each possible action is.

These estimates are stored in a table called the Q-table.

## 8. The Q-Table

Because the state space is `194,400` and there are `6` actions, the Q-table is a matrix of size `194,400 x 6`. 

At the beginning, the Q-table is filled with zeros because the robot knows nothing. During training, the robot updates this table based on experience.

## 9. The Q-Learning Update Rule

The Q-learning update formula is:

```text
Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max(Q(next_state, all_actions)) - Q(s, a)]
```

Where:

| Symbol                            | Meaning                                 |
| --------------------------------- | --------------------------------------- |
| `s`                               | Current state                           |
| `a`                               | Action taken                            |
| `reward`                          | Reward received after taking the action |
| `next_state`                      | State after the action                  |
| `alpha`                           | Learning rate                           |
| `gamma`                           | Discount factor                         |
| `max(Q(next_state, all_actions))` | Best estimated future value             |

## 10. Hyperparameters

The editable training settings are placed in:

```text
config.py
```

Current values:

| Setting                 | Value | Meaning                           |
| ----------------------- | ----: | --------------------------------- |
| `TOTAL_EPISODES`        | 50,000| Number of training episodes       |
| `MAX_STEPS_PER_EPISODE` |   400 | Maximum steps before timeout      |
| `LEARNING_RATE`         |   0.1 | How fast Q-values update          |
| `DISCOUNT_FACTOR`       |  0.99 | How much future rewards matter    |
| `EPSILON_START`         |   1.0 | Initial exploration probability   |
| `EPSILON_DECAY`         |0.9999 | Exploration reduction per episode |

*Note on EPSILON_DECAY*: Because the state space is so massive (194k states), the robot needs a very slow epsilon decay (0.9999) to ensure it explores enough states before it starts exploiting its knowledge.

## 11. Exploration vs Exploitation

A robot that only exploits from the beginning will not learn much. Exploration (random actions) allows the robot to accidentally discover successful deliveries. As epsilon decays, the robot gradually shifts from experimenting to using what it has learned.

## 12. Training Loop

At a high level, each training episode works like this:

```text
reset environment (randomize dropoff assignments)
start robot at depot

for each step:
    choose action using epsilon-greedy
    apply action to environment
    receive reward and next state
    update Q-table
    stop if all 4 packages are delivered or time runs out

decay epsilon
```

## 13. How Do We Know the Robot Is Learning?

The training script records metrics and generates charts:
- `rewards.png`: Total reward should increase as the robot stops hitting obstacles and learns to manage its inventory.
- `steps.png`: Steps should decrease as the robot finds optimal routing for 4 packages.
- `accuracy.png`: The success rate of delivering all 4 packages.

## 14. Why This Project Is a Good Reinforcement Learning Example

This project demonstrates how Q-learning can solve complex logistics problems that are difficult to hardcode. Writing a script to manually route a robot with a 2-package capacity to 4 random destinations, while avoiding obstacles, is a complex pathfinding problem. The RL agent solves this natively just by pursuing the reward.

## 15. Visualization

The environment is rendered with Pygame (`run.py`).

The visual design includes:
- Distinct colors for the 4 waiting packages (Gold, Teal, Pink, Lime).
- Matching colored dropoff markers that pulse dynamically.
- A detailed delivery robot that displays colored boxes on its chassis representing its currently carried inventory.

## 16. Running the Project

Train the robot and then run the visual demo:

```powershell
python main.py
```

Run training without opening the demo window:

```powershell
python main.py --skip-demo
```

Run only the saved trained policy demo:

```powershell
python main.py --demo-only
```

## 17. Final Interpretation

The Q-table is the robot's learned memory. For any of the 194,400 possible situations the robot might find itself in, the table holds the answer to the question: "What is the most profitable action to take next?" 

This captures the core idea of Q-learning: learn the value of actions through experience, then use those values to execute complex, multi-stage plans.
