from __future__ import annotations

import random

from model.sets.actions import Action
from model.sets.states import State
from rl.q_table import QTable


class EpsilonGreedyPolicy:
    def __init__(self, q_table: QTable, epsilon: float) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        self.q_table = q_table
        self.epsilon = epsilon

    def sample_action(self, state: State, rng: random.Random) -> Action:
        actions = self.q_table.actions(state)
        if not actions:
            raise ValueError(f"No actions available for state {state.name}.")
        if rng.random() < self.epsilon:
            return rng.choice(actions)
        return rng.choice(self.q_table.best_actions(state))

