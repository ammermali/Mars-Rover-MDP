from config.config_loader import MdpConfig, load_mdp_config
from model.sets.actions import Action
from model.sets.states import State
from model.distributions.sampler import sample_distribution
import random
from typing import Mapping

def transition_distribution(
    state: State,
    action: Action,
    mdp: MdpConfig | None = None,
) -> Mapping[State, float]:
    if state.is_final():
        return {state: 1.0}
    model = load_mdp_config() if mdp is None else mdp
    try:
        return model.transitions[state, action]
    except KeyError as error:
        raise ValueError("Transition not defined.") from error

def pr_transition(state: State, action: Action, new_state: State, mdp: MdpConfig | None = None) -> float:
    return transition_distribution(state, action, mdp).get(new_state, 0.0)

def sample_next_state(state: State, action: Action, rng: random.Random, mdp: MdpConfig | None = None) -> State:
    distribution = transition_distribution(state, action, mdp)
    return sample_distribution(distribution, rng)
