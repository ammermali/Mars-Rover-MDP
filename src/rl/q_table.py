from __future__ import annotations

from pathlib import Path

import yaml

from config.config_loader import MdpConfig
from model.sets.actions import Action
from model.sets.states import State


class QTable:
    def __init__(self, mdp: MdpConfig) -> None:
        self._actions = {
            state: tuple(action for action in Action if (state, action) in mdp.transitions)
            for state in State
            if not state.is_final()
        }
        self._values = {
            state: {action: 0.0 for action in actions}
            for state, actions in self._actions.items()
        }

    def actions(self, state: State) -> tuple[Action, ...]:
        return self._actions.get(state, ())

    def get(self, state: State, action: Action) -> float:
        return self._values[state][action]

    def set(self, state: State, action: Action, value: float) -> None:
        self._values[state][action] = value

    def max_value(self, state: State) -> float:
        actions = self.actions(state)
        if not actions:
            return 0.0
        return max(self.get(state, action) for action in actions)

    def best_actions(self, state: State) -> tuple[Action, ...]:
        actions = self.actions(state)
        if not actions:
            return ()
        best_value = self.max_value(state)
        return tuple(action for action in actions if self.get(state, action) == best_value)

    def greedy_policy_table(self) -> dict[State, dict[Action, float]]:
        table: dict[State, dict[Action, float]] = {}
        for state in self._actions:
            best_actions = self.best_actions(state)
            probability = 1.0 / len(best_actions)
            table[state] = {action: probability for action in best_actions}
        return table

    def epsilon_greedy_policy_table(self, epsilon: float) -> dict[State, dict[Action, float]]:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        table: dict[State, dict[Action, float]] = {}
        for state, actions in self._actions.items():
            best_actions = self.best_actions(state)
            exploration_probability = epsilon / len(actions)
            exploitation_probability = (1 - epsilon) / len(best_actions)
            table[state] = {
                action: exploration_probability + (exploitation_probability if action in best_actions else 0.0)
                for action in actions
            }
        return table

    def values_by_name(self) -> dict[str, dict[str, float]]:
        return {
            state.name: {action.name: value for action, value in actions.items()}
            for state, actions in self._values.items()
        }

    def save_greedy_policy(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_policy = {
            state.name: {action.name: probability for action, probability in actions.items()}
            for state, actions in self.greedy_policy_table().items()
        }
        with open(output_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(raw_policy, file, sort_keys=False)
