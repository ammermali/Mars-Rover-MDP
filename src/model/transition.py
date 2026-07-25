from dataclasses import dataclass
from model.sets.actions import Action
from model.sets.states import State

@dataclass(frozen=True)
class Transition:
    state: State
    action: Action
    reward: float
    next_state: State
    terminated: bool