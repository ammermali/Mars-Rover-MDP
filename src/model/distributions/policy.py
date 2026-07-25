from model.sets.actions import Action
from model.sets.states import State
from model.distributions.sampler import sample_distribution
import random
from pathlib import Path
from typing import Mapping
import yaml


class Policy:
    table: Mapping[State, Mapping[Action, float]]

    def __init__(self, table: Mapping[State, Mapping[Action, float]] | None = None) -> None:
        self.table = table

    # Load policy from yaml file
    def load_from_yaml(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as file:
            raw_policy = yaml.safe_load(file)
            self.table = {
                State[state_name] : {
                    Action[action_name] : probability
                    for action_name, probability in action_dist.items()
                }
                for state_name, action_dist in raw_policy.items()
            }

    # Sampling
    def sample_action(self, state: State, rng: random.Random) -> Action:
        try:
            action_distribution = self.table[state]
        except KeyError as err:
            raise ValueError("Policy not defined.") from err
        return sample_distribution(action_distribution, rng)

    def get(self):
        return self.table