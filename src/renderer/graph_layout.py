from config.config_loader import load_config
from model.sets.states import State

NODE_POSITIONS: dict[State, dict[str, float]] = load_config().visualization.node_positions
