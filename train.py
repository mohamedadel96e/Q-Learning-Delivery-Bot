"""Training module — Q-learning training loop, metrics, and plot generation."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np

import config
import delivery_env  # noqa: F401 — registers DeliveryBot-v0

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    """Return a greedy action, randomly breaking exact ties."""
    best_value = np.max(q_values)
    best_actions = np.flatnonzero(q_values == best_value)
    return int(rng.choice(best_actions))


def train(
    total_episodes: int = config.TOTAL_EPISODES,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    learning_rate: float = config.LEARNING_RATE,
    discount_factor: float = config.DISCOUNT_FACTOR,
    epsilon_start: float = config.EPSILON_START,
    epsilon_decay: float = config.EPSILON_DECAY,
    epsilon_min: float = config.EPSILON_MIN,
    seed: int = config.RANDOM_SEED,
    log_interval: int = config.TRAINING_LOG_INTERVAL,
    qtable_path: Path = config.QTABLE_PATH,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Train a tabular Q-learning policy and save the learned Q-table."""
    rng = np.random.default_rng(seed)
    env = gym.make(
        config.ENV_ID,
        max_steps=max_steps,
        max_episode_steps=max_steps,
    )
    env.action_space.seed(seed)

    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n), dtype=np.float32
    )
    epsilon = epsilon_start

    metrics: dict[str, list[float]] = {
        "rewards": [],
        "steps": [],
        "successes": [],
    }

    for episode in range(1, total_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        success = 0
        steps_taken = 0

        for step in range(1, max_steps + 1):
            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = greedy_action(q_table[state], rng)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            future_value = 0.0 if done else float(np.max(q_table[next_state]))
            td_target = reward + discount_factor * future_value
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            state = next_state
            total_reward += reward
            steps_taken = step

            if info.get("is_success", False):
                success = 1

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        metrics["rewards"].append(float(total_reward))
        metrics["steps"].append(float(steps_taken))
        metrics["successes"].append(float(success))

        if episode % log_interval == 0 or episode == total_episodes:
            window = min(log_interval, len(metrics["rewards"]))
            avg_reward = np.mean(metrics["rewards"][-window:])
            avg_steps = np.mean(metrics["steps"][-window:])
            success_rate = np.mean(metrics["successes"][-window:]) * 100
            print(
                f"Episode {episode:5d}/{total_episodes} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Avg Steps: {avg_steps:6.2f} | "
                f"Success Rate: {success_rate:6.2f}% | "
                f"Epsilon: {epsilon:.3f}"
            )

    np.save(qtable_path, q_table)
    env.close()
    print(f"Saved Q-table to {qtable_path}")

    return q_table, metrics


# ────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────

def rolling_average(
    values: list[float],
    window: int = config.PLOT_ROLLING_WINDOW,
) -> np.ndarray:
    """Return a same-length rolling average for smooth training plots."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array

    averages = np.empty_like(array, dtype=np.float32)
    for index in range(array.size):
        start = max(0, index - window + 1)
        averages[index] = np.mean(array[start : index + 1])
    return averages


def save_plot(
    values: list[float],
    title: str,
    ylabel: str,
    path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    smoothed = rolling_average(values, window=config.PLOT_ROLLING_WINDOW)
    episodes = np.arange(1, len(smoothed) + 1)

    plt.figure(figsize=config.PLOT_FIGURE_SIZE)
    plt.plot(episodes, smoothed, linewidth=2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=config.PLOT_DPI)
    plt.close()
    print(f"Saved {path}")


def save_training_plots(metrics: dict[str, list[float]]) -> None:
    w = config.PLOT_ROLLING_WINDOW
    save_plot(
        metrics["rewards"],
        f"DeliveryBot Training Reward (Rolling {w} Episodes)",
        "Total Reward",
        config.REWARDS_PLOT_PATH,
    )
    save_plot(
        metrics["steps"],
        f"DeliveryBot Steps Per Episode (Rolling {w} Episodes)",
        "Steps",
        config.STEPS_PLOT_PATH,
    )
    save_plot(
        metrics["successes"],
        f"DeliveryBot Success Rate (Rolling {w} Episodes)",
        "Success Rate",
        config.ACCURACY_PLOT_PATH,
        ylim=(-0.05, 1.05),
    )
