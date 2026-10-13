from config.config_loader import MdpConfig, load_mdp_config
from model.sets.states import State
from model.sets.actions import Action

def reward(state: State, action: Action, new_state: State, mdp: MdpConfig | None = None) -> float:
    model = load_mdp_config() if mdp is None else mdp
    return model.rewards.get((state, action, new_state), 0.0)
