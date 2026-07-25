from model.distributions.state_transition import transition_ditribution
from model.sets.states import State
from dataclasses import dataclass
from model.sets.actions import Action
from model.reward import reward
from model.distributions.sampler import sample_distribution
import random

@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    terminated: bool

class Environment:
    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng
        self.state = State.START # 100% initial state distribution on START

    def reset(self) -> State:
        self.state = State.START
        return self.state

    # For now, the model doesn't have any missng actions per-state.
    def available_actions(self) -> tuple[Action, ...]:
        if self.state.is_final():
            return ()
        return tuple(Action)

    def step(self, action: Action) -> StepResult:
        if self.state.is_final():
            raise RuntimeError("Cannot take action in final state.")
        if action not in self.available_actions():
            raise ValueError("Invalid action.")
        old_state = self.state
        distribution = transition_ditribution(state=old_state, action=action)
        new_state = sample_distribution(distribution = distribution, rng = self.rng)
        r = reward(state=old_state, action=action, new_state=new_state)
        self.state = new_state
        return StepResult(state=new_state, reward=r, terminated=new_state.is_final())