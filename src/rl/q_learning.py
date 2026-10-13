from __future__ import annotations

from dataclasses import dataclass

from model.sets.states import State
from model.transition import Transition
from rl.q_table import QTable


@dataclass
class QLearningObserver:
    q_table: QTable
    alpha: float
    gamma: float
    max_update_delta: float = 0.0

    def on_episode_started(self, episode_index: int) -> None:
        pass

    def on_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        current_value = self.q_table.get(transition.state, transition.action)
        bootstrap = 0.0 if transition.terminated else self.q_table.max_value(transition.next_state)
        target = transition.reward + self.gamma * bootstrap
        updated_value = current_value + self.alpha * (target - current_value)
        self.q_table.set(transition.state, transition.action, updated_value)
        self.max_update_delta = max(self.max_update_delta, abs(updated_value - current_value))

    def on_episode_finished(
        self,
        episode_index: int,
        total_reward: float,
        steps: int,
        terminated: bool,
        final_state: State,
    ) -> None:
        pass

    def consume_max_update_delta(self) -> float:
        delta = self.max_update_delta
        self.max_update_delta = 0.0
        return delta

