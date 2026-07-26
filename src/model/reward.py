from config.config_loader import load_mdp_config
from model.sets.states import State
from model.sets.actions import Action

REWARD: dict[tuple[State, Action, State], float] = load_mdp_config().rewards

def reward(state: State, action: Action, new_state: State) -> float:
    return REWARD.get((state, action, new_state), 0.0)
