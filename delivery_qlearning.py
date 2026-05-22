"""Train and demo a Q-learning delivery robot on DeliveryBot-v0."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np

import delivery_env  # noqa: F401 - importing registers DeliveryBot-v0

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =========================
# Editable project settings
# =========================

ENV_ID = "DeliveryBot-v0"

TOTAL_EPISODES = 8000
MAX_STEPS_PER_EPISODE = 150
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
RANDOM_SEED = 7
TRAINING_LOG_INTERVAL = 500

PLOT_ROLLING_WINDOW = 100
PLOT_FIGURE_SIZE = (10, 5)
PLOT_DPI = 150

DEMO_EPISODES = 10
DEMO_STEP_DELAY_MS = 250
DEMO_END_DELAY_MS = 900
DEMO_SEED_OFFSET = 900

QTABLE_PATH = Path("delivery_qtable.npy")
REWARDS_PLOT_PATH = Path("rewards.png")
STEPS_PLOT_PATH = Path("steps.png")
ACCURACY_PLOT_PATH = Path("accuracy.png")


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    """Return a greedy action, randomly breaking exact ties."""
    best_value = np.max(q_values)
    best_actions = np.flatnonzero(q_values == best_value)
    return int(rng.choice(best_actions))


def train(
    total_episodes: int = TOTAL_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    learning_rate: float = LEARNING_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    epsilon_start: float = EPSILON_START,
    epsilon_decay: float = EPSILON_DECAY,
    epsilon_min: float = EPSILON_MIN,
    seed: int = RANDOM_SEED,
    log_interval: int = TRAINING_LOG_INTERVAL,
    qtable_path: Path = QTABLE_PATH,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Train a tabular Q-learning policy and save the learned Q-table."""
    rng = np.random.default_rng(seed)
    env = gym.make(
        ENV_ID,
        max_steps=max_steps,
        max_episode_steps=max_steps,
    )
    env.action_space.seed(seed)

    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    epsilon = epsilon_start

    metrics = {
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


def rolling_average(
    values: list[float],
    window: int = PLOT_ROLLING_WINDOW,
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
    smoothed = rolling_average(values, window=PLOT_ROLLING_WINDOW)
    episodes = np.arange(1, len(smoothed) + 1)

    plt.figure(figsize=PLOT_FIGURE_SIZE)
    plt.plot(episodes, smoothed, linewidth=2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print(f"Saved {path}")


def save_training_plots(metrics: dict[str, list[float]]) -> None:
    save_plot(
        metrics["rewards"],
        f"DeliveryBot Training Reward (Rolling {PLOT_ROLLING_WINDOW} Episodes)",
        "Total Reward",
        REWARDS_PLOT_PATH,
    )
    save_plot(
        metrics["steps"],
        f"DeliveryBot Steps Per Episode (Rolling {PLOT_ROLLING_WINDOW} Episodes)",
        "Steps",
        STEPS_PLOT_PATH,
    )
    save_plot(
        metrics["successes"],
        f"DeliveryBot Success Rate (Rolling {PLOT_ROLLING_WINDOW} Episodes)",
        "Success Rate",
        ACCURACY_PLOT_PATH,
        ylim=(-0.05, 1.05),
    )


def run_demo(
    episodes: int = DEMO_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    qtable_path: Path = QTABLE_PATH,
    seed: int = RANDOM_SEED + DEMO_SEED_OFFSET,
    step_delay_ms: int = DEMO_STEP_DELAY_MS,
    end_delay_ms: int = DEMO_END_DELAY_MS,
) -> None:
    """Play trained DeliveryBot episodes with the Pygame renderer."""
    if not qtable_path.exists():
        raise FileNotFoundError(
            f"{qtable_path} does not exist yet. Train first or pass --demo-only "
            "after creating the Q-table."
        )

    import pygame

    q_table = np.load(qtable_path)
    rng = np.random.default_rng(seed)
    env = gym.make(
        ENV_ID,
        render_mode="human",
        max_steps=max_steps,
        max_episode_steps=max_steps,
    )

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        print(
            f"Demo {episode}: pickup {info['pickup_position']} -> "
            f"dropoff {info['dropoff_position']}"
        )

        for step in range(1, max_steps + 1):
            action = greedy_action(q_table[state], rng)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            pygame.time.wait(step_delay_ms)

            if terminated or truncated:
                status = "success" if info.get("is_success", False) else "timeout"
                print(
                    f"Demo {episode} finished with {status} in {step} steps "
                    f"(reward {total_reward})."
                )
                pygame.time.wait(end_delay_ms)
                break

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize a tabular Q-learning delivery robot."
    )
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=DEMO_EPISODES,
        help="Number of visual demo episodes to play.",
    )
    parser.add_argument(
        "--demo-delay-ms",
        type=int,
        default=DEMO_STEP_DELAY_MS,
        help="Delay between rendered demo steps in milliseconds.",
    )
    parser.add_argument(
        "--demo-end-delay-ms",
        type=int,
        default=DEMO_END_DELAY_MS,
        help="Pause after each completed demo episode in milliseconds.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Train and save plots without opening the Pygame demo window.",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Skip training and only replay delivery_qtable.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo_only:
        run_demo(
            episodes=args.demo_episodes,
            max_steps=args.max_steps,
            seed=args.seed + DEMO_SEED_OFFSET,
            step_delay_ms=args.demo_delay_ms,
            end_delay_ms=args.demo_end_delay_ms,
        )
        return

    _, metrics = train(
        total_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    save_training_plots(metrics)

    if not args.skip_demo:
        run_demo(
            episodes=args.demo_episodes,
            max_steps=args.max_steps,
            seed=args.seed + DEMO_SEED_OFFSET,
            step_delay_ms=args.demo_delay_ms,
            end_delay_ms=args.demo_end_delay_ms,
        )


if __name__ == "__main__":
    main()
