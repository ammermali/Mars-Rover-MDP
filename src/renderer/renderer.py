from typing import Protocol
from visualization.visualization_state import VisualizationState


class Renderer(Protocol):
    def render_episode_started(self, state: VisualizationState) -> None:
        ...

    def render_step(self, state: VisualizationState) -> None:
        ...

    def render_episode_finished(self, state: VisualizationState) -> None:
        ...

    def close(self) -> None:
        ...
