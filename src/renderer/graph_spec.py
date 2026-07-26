from dataclasses import dataclass
from model.sets.states import State
from model.sets.actions import Action
from model.reward import reward
from model.distributions.state_transition import TRANSITIONS


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

def build_graph_spec() -> GraphSpec:
    states = list(State)
    nodes = [GraphNode(state, state.is_final()) for state in states]
    edges: list[GraphEdge] = []
    for (state, action), distribution in TRANSITIONS.items():
        for next_state, probability in distribution.items():
            edges.append(GraphEdge(state, action, next_state, probability, reward(state, action, next_state)))
    return GraphSpec(nodes, edges)

