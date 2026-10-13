from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from model.sets.actions import Action
from model.sets.states import State
from model.transition import Transition
from rl.q_learning import QLearningObserver
from rl.q_table import QTable
from rl.training import IntervalMetrics


@dataclass(frozen=True)
class TrainingUpdate:
    episode: int
    total_episodes: int
    epsilon: float
    latest_reward: float | None
    completed: bool
    policy_table: Mapping[State, Mapping[Action, float]]
    q_values: Mapping[State, Mapping[Action, float]]
    convergence: tuple[IntervalMetrics, ...]


class TrainingUpdateSink(Protocol):
    def render_training_update(self, update: TrainingUpdate) -> None:
        ...


class LiveTrainingObserver(QLearningObserver):
    def __init__(
        self,
        q_table: QTable,
        alpha: float,
        gamma: float,
        sink: TrainingUpdateSink,
        total_episodes: int,
    ) -> None:
        super().__init__(q_table=q_table, alpha=alpha, gamma=gamma)
        self.sink = sink
        self.total_episodes = total_episodes
        self.episode = 0
        self.epsilon = 0.0
        self.latest_reward: float | None = None
        self.completed = False
        self.convergence: list[IntervalMetrics] = []
        self._publish()

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def on_episode_started(self, episode_index: int) -> None:
        self.episode = episode_index + 1
        self._publish()

    def on_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        super().on_step(episode_index, step_index, transition)
        self._publish()

    def on_episode_finished(
        self,
        episode_index: int,
        total_reward: float,
        steps: int,
        terminated: bool,
        final_state: State,
    ) -> None:
        self.latest_reward = total_reward
        self._publish()

    def on_interval_metrics(self, metrics: IntervalMetrics) -> None:
        self.convergence.append(metrics)
        self._publish()

    def mark_completed(self) -> None:
        self.completed = True
        self._publish()

    def _publish(self) -> None:
        self.sink.render_training_update(
            TrainingUpdate(
                episode=self.episode,
                total_episodes=self.total_episodes,
                epsilon=self.epsilon,
                latest_reward=self.latest_reward,
                completed=self.completed,
                policy_table=self.q_table.epsilon_greedy_policy_table(self.epsilon),
                q_values={
                    state: {action: self.q_table.get(state, action) for action in self.q_table.actions(state)}
                    for state in State
                    if self.q_table.actions(state)
                },
                convergence=tuple(self.convergence),
            )
        )
