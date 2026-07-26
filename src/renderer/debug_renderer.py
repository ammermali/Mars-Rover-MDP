from renderer.renderer import Renderer
from visualization.visualization_state import VisualizationState


# Renderer used only for the debug purposes.
class DebugRenderer(Renderer):
    def render_episode_started(self, state: VisualizationState) -> None:
        print(f"[render] episode={state.episode_index} start={state.current_state.name}")

    def render_step(self, state: VisualizationState) -> None:
        edge = state.current_edge
        if edge is None:
            return

        print(
            f"[render] episode={state.episode_index} "
            f"step={state.step_index} "
            f"{edge.source.name} --{edge.action.name}--> {edge.target.name} "
            f"reward={state.last_reward:.2f} "
            f"total={state.total_reward:.2f}"
        )

    def render_episode_finished(self, state: VisualizationState) -> None:
        print(
            f"[render] episode={state.episode_index} finished "
            f"state={state.current_state.name} "
            f"terminated={state.terminated} "
            f"total={state.total_reward:.2f}"
        )

    def close(self) -> None:
        pass