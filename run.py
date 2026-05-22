"""Running module — replay trained Q-table episodes with Pygame rendering."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

import config
import delivery_env  # noqa: F401 — registers DeliveryBot-v0
from train import greedy_action


def run_demo(
    episodes: int = config.DEMO_EPISODES,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    qtable_path: Path = config.QTABLE_PATH,
    seed: int = config.RANDOM_SEED + config.DEMO_SEED_OFFSET,
    step_delay_ms: int = config.DEMO_STEP_DELAY_MS,
    end_delay_ms: int = config.DEMO_END_DELAY_MS,
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
        config.ENV_ID,
        render_mode="human",
        max_steps=max_steps,
        max_episode_steps=max_steps,
    )

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        colors = {0: "red", 1: "blue", 2: "yellow"}
        print(
            f"Demo {episode}: pickup {info['pickup_position']} -> "
            f"dropoff {info['dropoff_position']} | "
            f"package: {colors.get(info['package_type'])}"
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
