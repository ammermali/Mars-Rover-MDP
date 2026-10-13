from dataclasses import dataclass
from model.sets.states import State
from model.sets.actions import Action
from config.config_loader import MdpConfig, load_mdp_config


@dataclass(frozen=True)
class GraphNode:
    state: State
    is_final: bool

@dataclass(frozen=True)
class GraphEdge:
    source: State
    action: Action
    target: State
    probability: float
    reward: float

@dataclass(frozen=True)
class GraphSpec:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

def build_graph_spec(mdp: MdpConfig | None = None) -> GraphSpec:
    model = load_mdp_config() if mdp is None else mdp
    states = list(State)
    nodes = [GraphNode(state, state.is_final()) for state in states]
    edges: list[GraphEdge] = []
    for (state, action), distribution in model.transitions.items():
        for next_state, probability in distribution.items():
            edges.append(GraphEdge(state, action, next_state, probability, model.rewards.get((state, action, next_state), 0.0)))
    return GraphSpec(nodes, edges)

