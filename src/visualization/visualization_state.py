from model.sets.actions import Action
from model.sets.states import State
from model.transition import Transition
from collections import Counter
from dataclasses import dataclass, field
from model.episode_observer import EpisodeObserver

# Represents an edge that has been moved over
@dataclass(frozen=True)
class EdgeKey:
    source: State
    action: Action
    target: State

@dataclass
class VisualizationState:
    start_state: State = State.START
    episode_index: int | None = None
    step_index: int | None = None
    previous_state: State | None = None
    current_state: State = State.START
    current_action: Action | None = None
    current_edge : EdgeKey | None = None
    total_reward: float = 0.0
    last_reward: float | None = None
    state_path: list[State] = field(default_factory=list)
    visited_states: Counter[State] = field(default_factory=Counter)
    visited_edges: Counter[EdgeKey] = field(default_factory=Counter)
    outcome_counts: Counter[State] = field(default_factory=Counter)
    terminated: bool = False

    def start_episode(self, episode_index: int) -> None:
        self.episode_index = episode_index
        self.step_index = None
        self.previous_state = None
        self.current_state = self.start_state
        self.current_action = None
        self.current_edge = None
        self.total_reward = 0.0
        self.last_reward = None
        self.state_path = [self.start_state]
        self.visited_states.clear()
        self.visited_edges.clear()
        self.visited_states[self.start_state] += 1
        self.terminated = False

    def record_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        self.episode_index = episode_index
        self.step_index = step_index
        self.previous_state = transition.state
        self.current_state = transition.next_state
        self.current_action = transition.action
        self.current_edge = EdgeKey(source=transition.state, action=transition.action, target=transition.next_state)
        self.total_reward += transition.reward
        self.last_reward = transition.reward
        self.state_path.append(transition.next_state)
        self.visited_states[self.current_state] += 1
        self.visited_edges[self.current_edge] += 1
        self.terminated = transition.terminated

    def finish_episode(self, episode_index: int, total_reward: float, steps: int, terminated: bool, final_state: State) -> None:
        self.episode_index = episode_index
        self.step_index = steps - 1 if steps > 0 else None
        self.current_state = final_state
        self.total_reward = total_reward
        self.terminated = terminated
        if terminated and final_state.is_final():
            self.outcome_counts[final_state] += 1
