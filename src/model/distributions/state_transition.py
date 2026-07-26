from config.config_loader import load_mdp_config
from model.sets.actions import Action
from model.sets.states import State
from model.distributions.sampler import sample_distribution
import random
from typing import Mapping

TRANSITIONS: dict[tuple[State, Action], dict[State, float]] = load_mdp_config().transitions

def transition_distribution(state: State, action: Action) -> Mapping[State, float]:
    if state.is_final():
        return {state: 1.0}
    try:
        return TRANSITIONS[state,action]
    # TODO: return 0.0 ??
    except KeyError as error:
        raise ValueError("Transition not defined.") from error

def pr_transition(state: State, action: Action, new_state: State) -> float:
    return transition_distribution(state, action).get(new_state, 0.0)

# sampling TODO: keep it?
def sample_next_state(state: State, action: Action, rng: random.Random) -> State:
    distribution = transition_distribution(state, action)
    return sample_distribution(distribution, rng)
