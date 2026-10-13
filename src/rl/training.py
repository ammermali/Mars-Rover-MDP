from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Callable

from config.config_loader import MdpConfig, RLConfig
from model.distributions.policy import ActionPolicy
from model.environment import Environment
from model.episode import EpisodeResult, run_episode
from model.episode_observer import CompositeEpisodeObserver, EpisodeObserver
from model.sets.states import State
from rl.epsilon_greedy import EpsilonGreedyPolicy
from rl.q_learning import QLearningObserver
from rl.q_table import QTable


@dataclass(frozen=True)
class IntervalMetrics:
    episode: int
    average_reward: float
    base_rate: float
    epsilon: float
    max_q_update_delta: float


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    average_reward: float
    base_rate: float
    terminal_rate: float


@dataclass(frozen=True)
class TrainingResult:
    metrics: list[IntervalMetrics]
    evaluation: EvaluationResult
    q_values: dict[str, dict[str, float]]


class QLearningTrainer:
    def __init__(self, mdp: MdpConfig, rl_config: RLConfig, seed: int, max_steps: int) -> None:
        self.mdp = mdp
        self.rl_config = rl_config
        self.seed = seed
        self.max_steps = max_steps

    def train(
        self,
        learner_factory: Callable[[QTable], QLearningObserver] | None = None,
        episode_observer: EpisodeObserver | None = None,
    ) -> tuple[QTable, TrainingResult]:
        rng = random.Random(self.seed)
        q_table = QTable(self.mdp)
        learner = (
            QLearningObserver(q_table, self.rl_config.alpha, self.rl_config.gamma)
            if learner_factory is None
            else learner_factory(q_table)
        )
        metrics: list[IntervalMetrics] = []
        reward_window: list[float] = []
        base_window = 0
        observer = learner if episode_observer is None else CompositeEpisodeObserver([learner, episode_observer])

        for episode_index in range(self.rl_config.training_episodes):
            epsilon = self.epsilon_for(episode_index)
            set_epsilon = getattr(learner, "set_epsilon", None)
            if set_epsilon is not None:
                set_epsilon(epsilon)
            policy = EpsilonGreedyPolicy(q_table, epsilon)
            result = run_episode(
                environment=Environment(self.mdp, rng),
                policy=policy,
                max_steps=self.max_steps,
                observer=observer,
                episode_index=episode_index,
            )
            reward_window.append(result.total_reward)
            if self.final_state(result) is State.BASE:
                base_window += 1

            is_interval_end = (episode_index + 1) % self.rl_config.metrics_interval == 0
            is_final_episode = episode_index + 1 == self.rl_config.training_episodes
            if is_interval_end or is_final_episode:
                metric = IntervalMetrics(
                    episode=episode_index + 1,
                    average_reward=sum(reward_window) / len(reward_window),
                    base_rate=base_window / len(reward_window),
                    epsilon=epsilon,
                    max_q_update_delta=learner.consume_max_update_delta(),
                )
                metrics.append(metric)
                on_interval_metrics = getattr(learner, "on_interval_metrics", None)
                if on_interval_metrics is not None:
                    on_interval_metrics(metric)
                reward_window = []
                base_window = 0

        evaluation = evaluate_policy(
            self.mdp,
            EpsilonGreedyPolicy(q_table, epsilon=0.0),
            episodes=self.rl_config.evaluation_episodes,
            seed=self.seed + 1,
            max_steps=self.max_steps,
        )
        return q_table, TrainingResult(metrics=metrics, evaluation=evaluation, q_values=q_table.values_by_name())

    def epsilon_for(self, episode_index: int) -> float:
        return max(
            self.rl_config.epsilon_min,
            self.rl_config.epsilon_start * self.rl_config.epsilon_decay**episode_index,
        )

    @staticmethod
    def final_state(result: EpisodeResult) -> State:
        return result.transitions[-1].next_state if result.transitions else State.START


def evaluate_policy(
    mdp: MdpConfig,
    policy: ActionPolicy,
    episodes: int,
    seed: int,
    max_steps: int,
) -> EvaluationResult:
    rng = random.Random(seed)
    total_reward = 0.0
    base_episodes = 0
    terminal_episodes = 0
    for episode_index in range(episodes):
        result = run_episode(Environment(mdp, rng), policy, max_steps, episode_index=episode_index)
        total_reward += result.total_reward
        final_state = QLearningTrainer.final_state(result)
        if final_state is State.BASE:
            base_episodes += 1
        if result.terminated:
            terminal_episodes += 1
    return EvaluationResult(
        episodes=episodes,
        average_reward=total_reward / episodes,
        base_rate=base_episodes / episodes,
        terminal_rate=terminal_episodes / episodes,
    )


def save_training_report(path: str | Path, result: TrainingResult, rl_config: RLConfig) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "q_learning",
        "hyperparameters": {
            "alpha": rl_config.alpha,
            "gamma": rl_config.gamma,
            "epsilon_start": rl_config.epsilon_start,
            "epsilon_min": rl_config.epsilon_min,
            "epsilon_decay": rl_config.epsilon_decay,
            "training_episodes": rl_config.training_episodes,
            "visual_training_episodes": rl_config.visual_training_episodes,
            "evaluation_episodes": rl_config.evaluation_episodes,
        },
        "metrics": [asdict(metric) for metric in result.metrics],
        "evaluation": asdict(result.evaluation),
        "q_values": result.q_values,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
