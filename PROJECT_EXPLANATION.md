# Autonomous Delivery Robot Using Q-Learning

## 1. Project Idea

This project simulates an autonomous delivery robot moving through a small urban environment. The robot starts at a depot, navigates a 10x10 city grid, picks up a package, and delivers it to the correct destination.

The goal is not only to make the robot move, but to make it learn a useful delivery strategy through reinforcement learning. Instead of manually programming every route, we let the robot explore the environment, make mistakes, receive rewards or penalties, and gradually discover better decisions.

## 2. The Central Question

Before writing code, we should ask:

> If a delivery robot is placed inside a city block, what information does it need in order to make a good decision?

At first, the answer might seem simple:

- Where is the robot?
- Where is the package?
- Where is the customer?
- Is the robot currently carrying the package?
- Which moves are allowed?
- Which moves are bad?
- What counts as success?

These questions naturally lead us to a reinforcement learning formulation.

The robot is an agent. The city grid is the environment. Every time the robot chooses an action, the environment responds with a new state and a reward. Over many episodes, the robot learns which actions usually lead to successful delivery.

## 3. Environment Design

The custom Gymnasium environment is implemented in:

```text
delivery_env.py
```

The environment is registered as:

```text
DeliveryBot-v0
```

### 3.1 City Grid

The city is represented as a 10x10 grid.

Each cell can represent part of the urban environment:

- Open road or sidewalk
- A building cell that blocks movement
- The robot depot
- A package pickup location
- A customer dropoff location

The robot moves one cell at a time. This keeps the problem simple enough for tabular Q-learning, while still capturing the core idea of urban navigation.

### 3.2 Buildings and Obstacles

Buildings are impassable. If the robot tries to move into a building, the move is rejected and the robot receives a penalty.

This is important because a delivery robot cannot simply move in a straight line to the destination. It has to learn paths around blocked areas.

So we can ask:

> What is the difference between a short route and a valid route?

The shortest route geometrically may pass through buildings. The fastest valid route must respect the structure of the city.

### 3.3 Pickup and Dropoff Locations

At the start of every episode, the environment randomly chooses:

- One pickup location from a predefined set
- One dropoff location from a predefined set

This makes the task more interesting than memorizing one route. The robot must learn a reusable policy for multiple delivery jobs.

The current pickup locations are:

```text
(1, 1), (1, 8), (8, 1), (8, 8)
```

The current dropoff locations are:

```text
(0, 5), (5, 0), (5, 9), (9, 5)
```

### 3.4 Depot

The robot starts from a fixed depot:

```text
(0, 0)
```

This is realistic for a delivery robot because real delivery systems usually begin from a depot, warehouse, or charging station.

It also makes training more reliable because the robot is always solving the same kind of mission:

```text
depot -> pickup -> dropoff
```

## 4. State Space

The state tells the robot what situation it is currently in.

A weak state representation would only include:

```text
robot x-position
robot y-position
carrying package or not
```

But here is the problem:

> If the pickup and dropoff locations are randomized, can the robot choose correctly without knowing which pickup and dropoff are active?

No. The same robot position might require different actions depending on which package and destination were selected.

For example, standing at `(5, 5)` is not enough information. If the package is north, the robot should move north. If the package is south, it should move south.

So the environment encodes a richer discrete state:

```text
robot x-position
robot y-position
whether the robot is carrying the package
active pickup index
active dropoff index
```

This keeps the state space discrete, but also preserves the information needed for learning.

The state is encoded as one integer so it can be used directly as a row index in the Q-table.

### Why This Matters

This is a key reinforcement learning idea: the state must contain enough information for the agent to make a good decision.

If the state hides important information, the agent sees two different situations as identical. That makes learning unstable because the same state might sometimes require one action and sometimes another.

This is why the pickup and dropoff IDs are included.

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

This action design creates a useful question:

> Should pickup and dropoff be automatic, or should the robot choose them?

In this project, pickup and dropoff are explicit actions. That means the robot must learn not only where to move, but also when to perform task-specific actions.

This is closer to a real decision-making problem. A robot must know when it has reached the package and when it has reached the customer.

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

If every valid movement has a reward of `0`, the robot may still learn to deliver the package, but it has no strong reason to prefer shorter routes.

The `-1` step penalty asks:

> Can you solve the delivery using fewer moves?

This encourages the fastest route, not just any successful route.

### Why Penalize Illegal Actions?

Illegal actions include:

- Trying to move into a building
- Trying to pick up a package in the wrong place
- Trying to drop off when not carrying anything
- Trying to drop off at the wrong destination

The `-10` penalty teaches the robot that these actions waste time and should be avoided.

### Why Reward Pickup and Dropoff Separately?

The final delivery reward is large, but pickup also receives a positive reward.

This helps the robot understand the task in two stages:

```text
Stage 1: Find and pick up the package
Stage 2: Carry it to the correct dropoff point
```

Without a pickup reward, the robot would only receive a major positive signal at the very end. That can make learning slower, especially early in training.

## 7. What Is Q-Learning?

Q-learning is a reinforcement learning algorithm that learns the value of taking an action in a state.

The main idea is:

> For each state, estimate how good each possible action is.

These estimates are stored in a table called the Q-table.

Each row represents a state. Each column represents an action.

For this project:

```text
Q-table[state, action] = expected future value of taking that action in that state
```

At the beginning, the Q-table is filled with zeros because the robot knows nothing.

During training, the robot updates this table based on experience.

## 8. The Q-Table

The Q-table is created in:

```text
delivery_qlearning.py
```

The table shape is:

```text
number of states x number of actions
```

Since there are 6 actions, every state has 6 Q-values.

Conceptually, one row might look like this:

| State |   Up | Down | Left | Right | Pickup | Dropoff |
| ----: | ---: | ---: | ---: | ----: | -----: | ------: |
|  1420 | -3.2 | 18.5 | -7.1 |  12.4 |  -10.0 |   -10.0 |

The robot chooses the action with the highest Q-value when it wants to exploit what it has learned.

In the example above, action `Down` has the highest value, so the robot would choose `Down`.

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

Now let us think through this formula rather than memorize it.

### 9.1 What Does the Robot Already Believe?

The robot has an old estimate:

```text
Q(s, a)
```

This is what the robot previously believed about taking action `a` in state `s`.

### 9.2 What Did the Robot Just Learn?

After taking the action, the robot receives:

```text
reward
```

It also sees the next state and asks:

> From this next state, what is the best action I could take?

That is:

```text
max(Q(next_state, all_actions))
```

So the new target estimate becomes:

```text
reward + gamma * best future value
```

### 9.3 Why Not Replace the Old Value Immediately?

The robot does not completely overwrite its old belief. Instead, it moves gradually toward the new estimate.

That is controlled by:

```text
alpha
```

In this project:

```text
LEARNING_RATE = 0.1
```

This means each update changes the old estimate by 10% of the new error.

## 10. Hyperparameters

The editable training settings are placed at the top of:

```text
delivery_qlearning.py
```

Current values:

| Setting                 | Value | Meaning                           |
| ----------------------- | ----: | --------------------------------- |
| `TOTAL_EPISODES`        |  8000 | Number of training episodes       |
| `MAX_STEPS_PER_EPISODE` |   150 | Maximum steps before timeout      |
| `LEARNING_RATE`         |   0.1 | How fast Q-values update          |
| `DISCOUNT_FACTOR`       |  0.99 | How much future rewards matter    |
| `EPSILON_START`         |   1.0 | Initial exploration probability   |
| `EPSILON_DECAY`         | 0.999 | Exploration reduction per episode |
| `EPSILON_MIN`           |  0.01 | Minimum exploration probability   |
| `DEMO_EPISODES`         |    10 | Number of visual demo episodes    |
| `DEMO_STEP_DELAY_MS`    |   250 | Delay between demo steps          |

## 11. Exploration vs Exploitation

A robot that only exploits from the beginning will not learn much, because its Q-table starts empty.

So early in training, the robot explores.

Exploration means:

```text
Choose a random action
```

Exploitation means:

```text
Choose the action with the highest Q-value
```

The epsilon-greedy strategy controls this balance.

At the start:

```text
epsilon = 1.0
```

This means the robot mostly explores.

After every episode:

```text
epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
```

Over time, epsilon decreases. The robot gradually shifts from experimenting to using what it has learned.

This mirrors a natural learning process:

1. Try many actions early.
2. Notice which actions lead to better outcomes.
3. Trust the learned behavior more over time.

## 12. Training Loop

At a high level, each training episode works like this:

```text
reset environment
choose random pickup and dropoff
start robot at depot

for each step:
    choose action using epsilon-greedy
    apply action to environment
    receive reward and next state
    update Q-table
    stop if delivery succeeds or time runs out

decay epsilon
record reward, steps, and success
```

The training script records:

- Total reward per episode
- Steps taken per episode
- Whether the delivery succeeded

Every 500 episodes, it prints average metrics so we can see whether learning is improving.

## 13. How Do We Know the Robot Is Learning?

We expect three trends:

### Reward Should Increase

At first, the robot makes many mistakes, hits buildings, and performs illegal actions. Rewards are low.

As learning improves, rewards increase because the robot reaches pickup and dropoff more efficiently.

### Steps Should Decrease

Early episodes may reach the step limit.

Later episodes should finish faster because the robot has learned shorter valid routes.

### Success Rate Should Increase

The most important metric is whether the package reaches the correct destination.

The project saves:

```text
rewards.png
steps.png
accuracy.png
```

These plots use a rolling average to make the learning trend easier to read.

## 14. Why This Project Is a Good Reinforcement Learning Example

This project is small enough to understand, but rich enough to demonstrate important RL ideas:

- State design matters.
- Reward design shapes behavior.
- Exploration is necessary.
- A shortest path is not always a valid path.
- A learned policy can solve multiple randomized tasks.
- Visualization makes training behavior easier to interpret.

The project also shows the difference between pathfinding and reinforcement learning.

A traditional pathfinding algorithm like A\* computes a route from a known start to a known goal. Q-learning instead learns from repeated interaction. Once trained, the robot can react to the encoded delivery task by choosing actions from its learned Q-table.

## 15. Visualization

The environment is rendered with Pygame.

The visual design includes:

- Roads and sidewalks
- Crosswalks
- Buildings with windows and shadows
- Depot marker
- Package marker
- Dropoff target
- Streetlights
- Planters
- Traffic signals
- A detailed delivery robot

The visualization is not just decorative. It helps us inspect whether the learned behavior makes sense.

When watching the demo, ask:

> Does the robot go to the package first?

> Does it avoid buildings?

> Does it take a reasonably direct route?

> Does it only drop off after pickup?

These questions help connect the learned policy to the project objective.

## 16. Running the Project

Train the robot and then run the visual demo:

```powershell
venv\Scripts\python.exe delivery_qlearning.py
```

Run training without opening the demo window:

```powershell
venv\Scripts\python.exe delivery_qlearning.py --skip-demo
```

Run only the saved trained policy demo:

```powershell
venv\Scripts\python.exe delivery_qlearning.py --demo-only
```

Run more than 10 demos:

```powershell
venv\Scripts\python.exe delivery_qlearning.py --demo-only --demo-episodes 15
```

## 17. Final Interpretation

The robot is not given a hardcoded route. It learns a policy.

That policy answers this question for every state:

> Given where I am, whether I have the package, and which delivery job is active, what action should I take next?

The Q-table is the robot's learned memory of those answers.

At the beginning, the table is empty. Through trial, reward, penalty, and correction, the table becomes a map of good decisions.

This is the core idea of Q-learning:

> Learn the value of actions through experience, then use those values to make better decisions.

In this project, that idea becomes visible as a delivery robot that starts uncertain, improves over thousands of episodes, and eventually performs successful urban deliveries with fast, reliable routes.

## 18. Possible Extensions

Good next questions for improving the project:

- What happens if pedestrians become moving obstacles?
- Should the robot receive extra penalty for risky road crossings?
- Can the environment include traffic lights that change over time?
- Could the state include battery level?
- Could multiple packages be delivered in one episode?
- How would Q-learning compare with A\* or SARSA?
- At what grid size does tabular Q-learning become too large?

These extensions point toward a more realistic robotics simulation, but the current project already captures the main reinforcement learning pipeline from environment design to training, evaluation, and visualization.
