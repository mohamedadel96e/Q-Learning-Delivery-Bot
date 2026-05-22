"""Centralized configuration for the Q-Learning Delivery Bot project.

All tunable hyperparameters, reward values, file paths, and display
settings live here so they can be adjusted in one place without
touching the training, environment, or demo code.
"""

from __future__ import annotations

from pathlib import Path


# ──────────────────────────────────────
# Environment
# ──────────────────────────────────────
ENV_ID = "DeliveryBot-v0"
GRID_SIZE = 10
MAX_STEPS_PER_EPISODE = 400
NUM_PACKAGES = 4
MAX_CARRY = 2

# ──────────────────────────────────────
# Reward shaping
# ──────────────────────────────────────
STEP_REWARD = -1
ILLEGAL_REWARD = -10
PICKUP_REWARD = 20
DROPOFF_REWARD = 100

# ──────────────────────────────────────
# Q-Learning hyperparameters
# ──────────────────────────────────────
TOTAL_EPISODES = 50_000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.01
RANDOM_SEED = 7
TRAINING_LOG_INTERVAL = 500

# ──────────────────────────────────────
# Plotting
# ──────────────────────────────────────
PLOT_ROLLING_WINDOW = 100
PLOT_FIGURE_SIZE = (10, 5)
PLOT_DPI = 150

# ──────────────────────────────────────
# Demo / episode runner
# ──────────────────────────────────────
DEMO_EPISODES = 10
DEMO_STEP_DELAY_MS = 250
DEMO_END_DELAY_MS = 900
DEMO_SEED_OFFSET = 900

# ──────────────────────────────────────
# File paths
# ──────────────────────────────────────
QTABLE_PATH = Path("delivery_qtable.npy")
REWARDS_PLOT_PATH = Path("rewards.png")
STEPS_PLOT_PATH = Path("steps.png")
ACCURACY_PLOT_PATH = Path("accuracy.png")
