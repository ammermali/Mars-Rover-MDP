from model.sets.states import State
from model.sets.actions import Action

# Reward function
REWARD: dict[tuple[State, Action, State], float] = {
    (State.START, Action.LEFT, State.SITE_A) : -1.0,
    (State.START, Action.LEFT, State.IMMOBILE) : -3.0,
    (State.START, Action.RIGHT, State.BASE) : 10.0,
    (State.START, Action.RIGHT, State.DESTROYED) : -10.0,
    (State.SITE_A, Action.LEFT, State.START) : -1.0,
    (State.SITE_A, Action.RIGHT, State.SITE_B) : -1.0,
    (State.SITE_A, Action.RIGHT, State.IMMOBILE) : -3.0,
    (State.SITE_B, Action.LEFT, State.SITE_A) : -1.0,
    (State.SITE_B, Action.RIGHT, State.BASE): 10.0
}

def reward(state: State, action: Action, new_state: State) -> float:
    return REWARD.get((state, action, new_state), 0.0)