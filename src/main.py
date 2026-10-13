from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random
from uuid import uuid4

from config.config_loader import AppConfig, load_config
from logger.episode_observer import LoggingEpisodeObserver
from logger.logger_facade import LoggerFacade
from logger.sinks import CompositeSink, ConsoleSink, JsonFileSink
from model.distributions.policy import Policy
from model.environment import Environment
from model.episode import run_episode
from model.episode_observer import CompositeEpisodeObserver
from renderer.graph_spec import build_graph_spec
from rl.training import QLearningTrainer, evaluate_policy, save_training_report
from rl.live_training_observer import LiveTrainingObserver
from visualization.control_state import VisualizationControlState
from visualization.live_visualization_observer import LiveVisualizationObserver
from visualization.visualization_state import VisualizationState


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    if arguments.mode == "train":
        run_training(config)
    elif arguments.mode == "evaluate":
        run_evaluation(config)
    elif arguments.mode == "demo":
        run_demo(config)
    else:
        run_training_demo(config)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and inspect the Mars Rover Q-learning agent.")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "demo", "train-demo"),
        default="train",
        help="train, evaluate, display a saved policy, or train with live visualization",
    )
    parser.add_argument("--config", type=Path, default=None, help="path to the YAML configuration file")
    return parser.parse_args()


def run_training(config: AppConfig) -> None:
    trainer = QLearningTrainer(config.mdp, config.rl, config.experiment.seed, config.experiment.max_steps)
    q_table, result = trainer.train()
    q_table.save_greedy_policy(config.rl.policy_output_path)
    save_training_report(config.rl.report_output_path, result, config.rl)
    print("Q-learning training complete.")
    print(f"Saved policy: {config.rl.policy_output_path}")
    print(f"Saved report: {config.rl.report_output_path}")
    print(
        "Evaluation: "
        f"average_reward={result.evaluation.average_reward:.3f}, "
        f"base_rate={result.evaluation.base_rate:.3f}, "
        f"terminal_rate={result.evaluation.terminal_rate:.3f}"
    )


def run_evaluation(config: AppConfig) -> None:
    policy = load_learned_policy(config)
    result = evaluate_policy(
        config.mdp,
        policy,
        episodes=config.rl.evaluation_episodes,
        seed=config.experiment.seed + 1,
        max_steps=config.experiment.max_steps,
    )
    print(
        "Evaluation: "
        f"average_reward={result.average_reward:.3f}, "
        f"base_rate={result.base_rate:.3f}, "
        f"terminal_rate={result.terminal_rate:.3f}"
    )


def run_demo(config: AppConfig) -> None:
    from renderer.cytoscape_renderer import CytoscapeRenderer

    policy = load_learned_policy(config)
    control_state = VisualizationControlState(config.visualization.step_delay_seconds)
    renderer = CytoscapeRenderer(
        graph_spec=build_graph_spec(config.mdp),
        policy_table=policy.get(),
        control_state=control_state,
        host=config.visualization.host,
        port=config.visualization.port,
        poll_interval_ms=config.visualization.poll_interval_ms,
        start_label="Start simulation",
    )
    logger = build_logger(config)
    try:
        renderer.start()
        control_state.wait_for_simulation_start()
        logger.run_started(
            seed=config.experiment.seed,
            num_episodes=config.experiment.num_episodes,
            max_steps=config.experiment.max_steps,
            policy_path=config.rl.policy_output_path,
        )
        observer = CompositeEpisodeObserver(
            [
                LoggingEpisodeObserver(logger),
                LiveVisualizationObserver(VisualizationState(), control_state, renderer),
            ]
        )
        rng = random.Random(config.experiment.seed)
        total_reward = 0.0
        terminal_episodes = 0
        for episode_index in range(config.experiment.num_episodes):
            result = run_episode(
                Environment(config.mdp, rng),
                policy,
                config.experiment.max_steps,
                observer,
                episode_index,
            )
            total_reward += result.total_reward
            terminal_episodes += int(result.terminated)
        logger.run_finished(
            episodes=config.experiment.num_episodes,
            terminal_episodes=terminal_episodes,
            truncated_episodes=config.experiment.num_episodes - terminal_episodes,
            average_reward=total_reward / config.experiment.num_episodes,
            total_reward=total_reward,
        )
    except Exception as error:
        logger.error(error)
        raise
    finally:
        logger.close()
        renderer.close()


def run_training_demo(config: AppConfig) -> None:
    from renderer.cytoscape_renderer import CytoscapeRenderer

    visual_rl_config = replace(config.rl, training_episodes=config.rl.visual_training_episodes)
    control_state = VisualizationControlState(config.visualization.step_delay_seconds)
    renderer = CytoscapeRenderer(
        graph_spec=build_graph_spec(config.mdp),
        control_state=control_state,
        host=config.visualization.host,
        port=config.visualization.port,
        poll_interval_ms=config.visualization.poll_interval_ms,
        start_label="Start visual training",
    )
    live_observer: LiveTrainingObserver | None = None
    try:
        renderer.start()
        control_state.wait_for_simulation_start()
        trainer = QLearningTrainer(
            config.mdp,
            visual_rl_config,
            config.experiment.seed,
            config.experiment.max_steps,
        )

        def create_live_observer(q_table):
            nonlocal live_observer
            live_observer = LiveTrainingObserver(
                q_table=q_table,
                alpha=visual_rl_config.alpha,
                gamma=visual_rl_config.gamma,
                sink=renderer,
                total_episodes=visual_rl_config.training_episodes,
            )
            return live_observer

        q_table, result = trainer.train(
            learner_factory=create_live_observer,
            episode_observer=LiveVisualizationObserver(VisualizationState(), control_state, renderer),
        )
        q_table.save_greedy_policy(visual_rl_config.policy_output_path)
        save_training_report(visual_rl_config.report_output_path, result, visual_rl_config)
        if live_observer is not None:
            live_observer.mark_completed()
        print("Visual Q-learning training complete.")
        print(f"Saved policy: {visual_rl_config.policy_output_path}")
    finally:
        renderer.close()


def load_learned_policy(config: AppConfig) -> Policy:
    policy_path = config.rl.policy_output_path
    if not policy_path.exists():
        raise FileNotFoundError(f"No learned policy found at {policy_path}. Run with --mode train first.")
    policy = Policy()
    policy.load_from_yaml(policy_path)
    return policy


def build_logger(config: AppConfig) -> LoggerFacade:
    run_id = str(uuid4())
    sink = CompositeSink(
        [
            ConsoleSink(include_steps=False),
            JsonFileSink(config.logging.directory / f"{run_id}.json"),
        ]
    )
    return LoggerFacade(sink=sink, run_id=run_id)


if __name__ == "__main__":
    main()
