"""CLI entry point — dispatches to the training and running modules."""

from __future__ import annotations

import argparse

import config
from run import run_demo
from train import save_training_plots, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize a tabular Q-learning delivery robot.",
    )
    parser.add_argument("--episodes", type=int, default=config.TOTAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=config.DEMO_EPISODES,
        help="Number of visual demo episodes to play.",
    )
    parser.add_argument(
        "--demo-delay-ms",
        type=int,
        default=config.DEMO_STEP_DELAY_MS,
        help="Delay between rendered demo steps in milliseconds.",
    )
    parser.add_argument(
        "--demo-end-delay-ms",
        type=int,
        default=config.DEMO_END_DELAY_MS,
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
            seed=args.seed + config.DEMO_SEED_OFFSET,
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
            seed=args.seed + config.DEMO_SEED_OFFSET,
            step_delay_ms=args.demo_delay_ms,
            end_delay_ms=args.demo_end_delay_ms,
        )


if __name__ == "__main__":
    main()
