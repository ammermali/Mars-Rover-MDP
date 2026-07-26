from dataclasses import dataclass
from time import sleep
from model.sets.states import State
from model.transition import Transition
from visualization.control_state import VisualizationControlState
from visualization.visualization_state import VisualizationState
from model.episode_observer import EpisodeObserver
from renderer.renderer import Renderer


@dataclass
class LiveVisualizationObserver(EpisodeObserver):
    state: VisualizationState
    control_state: VisualizationControlState
    renderer: Renderer | None

    def __init__(self, state: VisualizationState | None = None, control_state: VisualizationControlState | None = None, renderer: Renderer | None = None) -> None:
        self.renderer = renderer
        self.state = state or VisualizationState()
        self.control_state = control_state or VisualizationControlState()

    def on_episode_started(self, episode_index: int) -> None:
        self.state.start_episode(episode_index)
        if self.renderer is not None:
            self.renderer.render_episode_started(self.state)

    def on_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        self.state.record_step(episode_index, step_index, transition)
        if self.renderer is not None:
            self.renderer.render_step(self.state)
        step_delay_seconds = self.control_state.get_step_delay_seconds()
        if step_delay_seconds > 0:
            sleep(step_delay_seconds)

    def on_episode_finished(self, episode_index: int, total_reward: float, steps: int, terminated: bool, final_state: State) -> None:
        self.state.finish_episode(episode_index, total_reward, steps, terminated, final_state)
        if self.renderer is not None:
            self.renderer.render_episode_finished(self.state)

