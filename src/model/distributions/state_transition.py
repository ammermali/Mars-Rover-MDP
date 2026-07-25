from model.sets.actions import Action
from model.sets.states import State
from model.distributions.sampler import sample_distribution
import random
from typing import Mapping

# Transition probabilities
TRANSITIONS: dict[tuple[State, Action], dict[State, float]] = {
    (State.START, Action.LEFT): {
        State.SITE_A : 0.9,
        State.IMMOBILE: 0.1
    },
    (State.START, Action.RIGHT): {
        State.BASE: 0.5,
        State.DESTROYED: 0.5
    },
    (State.SITE_A, Action.LEFT): {
        State.START: 1.0
    },
    (State.SITE_A, Action.RIGHT): {
        State.SITE_B: 0.8,
        State.IMMOBILE: 0.2
    },
    (State.SITE_B, Action.LEFT): {
        State.SITE_A: 1.0
    },
    (State.SITE_B, Action.RIGHT): {
        State.BASE: 1.0
    }
}

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