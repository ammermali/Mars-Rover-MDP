from dataclasses import dataclass
import random

from config.config_loader import MdpConfig, load_mdp_config
from model.distributions.sampler import sample_distribution
from model.sets.actions import Action
from model.sets.states import State

@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    terminated: bool

class Environment:
    def __init__(self, mdp: MdpConfig | None = None, rng: random.Random | None = None) -> None:
        self.mdp = load_mdp_config() if mdp is None else mdp
        self.rng = random.Random() if rng is None else rng
        self.state = State.START

    def reset(self) -> State:
        self.state = State.START
        return self.state

    def available_actions(self, state: State | None = None) -> tuple[Action, ...]:
        current_state = self.state if state is None else state
        if current_state.is_final():
            return ()
        return tuple(
            action
            for action in Action
            if (current_state, action) in self.mdp.transitions
        )

    def step(self, action: Action) -> StepResult:
        if self.state.is_final():
            raise RuntimeError("Cannot take action in final state.")
        if action not in self.available_actions():
            raise ValueError("Invalid action.")
        old_state = self.state
        try:
            distribution = self.mdp.transitions[old_state, action]
        except KeyError as error:
            raise ValueError("Transition not defined.") from error
        new_state = sample_distribution(distribution=distribution, rng=self.rng)
        r = self.mdp.rewards.get((old_state, action, new_state), 0.0)
        self.state = new_state
        return StepResult(state=new_state, reward=r, terminated=new_state.is_final())
