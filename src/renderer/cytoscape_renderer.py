from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Mapping
from dash import Dash, Input, Output, dcc, html
import dash_cytoscape as cyto
from renderer.graph_layout import NODE_POSITIONS
from renderer.graph_spec import GraphEdge, GraphSpec, build_graph_spec
from renderer.renderer import Renderer
from visualization.control_state import VisualizationControlState
from visualization.visualization_state import EdgeKey, VisualizationState
from model.sets.actions import Action
from model.sets.states import State

@dataclass(frozen=True)
class RendererSnapshot:
    episode_index: int | None = None
    step_index: int | None = None
    current_state: State | None = None
    current_edge: EdgeKey | None = None
    total_reward: float = 0.0
    last_reward: float | None = None
    terminated: bool = False
    state_path: list[State] = field(default_factory=list)
    visited_states: dict[State, int] = field(default_factory=dict)
    visited_edges: dict[EdgeKey, int] = field(default_factory=dict)
    outcome_counts: dict[State, int] = field(default_factory=dict)

class RenderStateStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = RendererSnapshot()

    def update(self, state: VisualizationState) -> None:
        with self._lock:
            self._snapshot = RendererSnapshot(
                episode_index=state.episode_index,
                step_index=state.step_index,
                current_state=state.current_state,
                current_edge=state.current_edge,
                total_reward=state.total_reward,
                last_reward=state.last_reward,
                terminated=state.terminated,
                state_path=list(state.state_path),
                visited_states=dict(state.visited_states),
                visited_edges=dict(state.visited_edges),
                outcome_counts=dict(state.outcome_counts)
            )

    def snapshot(self) -> RendererSnapshot:
        with self._lock:
            return self._snapshot


class CytoscapeRenderer(Renderer):
    def __init__(
        self,
        graph_spec: GraphSpec | None = None,
        policy_table: Mapping[State, Mapping[Action, float]] | None = None,
        control_state: VisualizationControlState | None = None,
        host: str = "127.0.0.1",
        port: int = 8050,
        poll_interval_ms: int = 100
    ) -> None:
        self.graph_spec = graph_spec or build_graph_spec()
        self.policy_table = policy_table
        self.control_state = control_state or VisualizationControlState()
        self.host = host
        self.port = port
        self.poll_interval_ms = poll_interval_ms
        self.store = RenderStateStore()
        self.app = self._build_app()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=lambda: self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False), daemon=False)
        self._thread.start()

    def render_episode_started(self, state: VisualizationState) -> None:
        self.store.update(state)

    def render_step(self, state: VisualizationState) -> None:
        self.store.update(state)

    def render_episode_finished(self, state: VisualizationState) -> None:
        self.store.update(state)

    def close(self) -> None:
        pass

    def _build_app(self) -> Dash:
        app = Dash(__name__)
        app.layout = html.Div(
            [
                dcc.Interval(id="render-tick", interval=self.poll_interval_ms, n_intervals=0),
                html.Div(
                    [
                        html.Div("Mars Rover MDP", className="title"),
                        html.Div(
                            [
                                html.Div(id="simulation-control-status"),
                                html.Button(
                                    "Start simulation",
                                    id="start-simulation-button",
                                    n_clicks=0
                                )
                            ],
                            className="run-controls",
                            style={"flexDirection": "column", "gap": "8px", "marginTop": "16px", "marginBottom": "16px", "alignItems": "center", "display": "flex", "justifyContent": "center"}
                        ),
                        html.Div(
                            [
                                html.Div("CURRENT POLICY:", className="section-title"),
                                html.Div(self._build_policy_matrix(), className="policy-table")
                            ],
                            className="policy-panel",
                            style={"marginBottom": "16px"}
                        ),
                        html.Div(
                            [
                                html.Div(id="step-delay-value"),
                                dcc.Slider(
                                    id="step-delay-slider",
                                    min=0.01,
                                    max=1.0,
                                    step=0.01,
                                    value=self.control_state.get_step_delay_seconds(),
                                    marks={
                                        0.1: "0.1s",
                                        0.5: "0.5s",
                                        1.0: "1s",
                                        2.0: "2s"
                                    }
                                )
                            ],
                            className="controls", style={"marginBottom": "16px", "gap": "8px"}
                        ),
                        html.Div(id="status-panel", className="status"),
                        html.Div(
                            [
                                html.Div("Visited states", className="section-title"),
                                html.Div(id="path-panel", className="path-box")
                            ],
                            className="path-panel"
                        ),
                        html.Div(
                            [
                                html.Div("Run outcomes", className="section-title"),
                                html.Div(id="outcome-panel", className="outcomes")
                            ],
                            className="outcome-panel"
                        )
                    ],
                    className="sidebar"
                ),
                cyto.Cytoscape(
                    id="mdp-graph",
                    elements=self._build_elements(self.store.snapshot()),
                    stylesheet=self._stylesheet(),
                    layout={"name": "preset", "fit": True, "padding": 40},
                    style={
                        "width": "100%",
                        "height": "720px",
                        "background": "#000000" #TODO: change background color (from config file)?
                    }
                )
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "320px 1fr",
                "gap": "16px",
                "background": "#0b1020", #TODO: change background color (from config file)?
                "color": "#e5e7eb",
                "minHeight": "100vh",
                "padding": "18px",
                "fontFamily": "Inter, Segoe UI, sans-serif"
            }
        )

        @app.callback(
            Output("mdp-graph", "elements"),
            Output("status-panel", "children"),
            Output("path-panel", "children"),
            Output("outcome-panel", "children"),
            Input("render-tick", "n_intervals")
        )
        def refresh(_: int):
            snapshot = self.store.snapshot()
            return (
                self._build_elements(snapshot),
                self._build_status(snapshot),
                self._build_state_path(snapshot),
                self._build_outcome_counts(snapshot)
            )

        @app.callback(Output("step-delay-value", "children"), Input("step-delay-slider", "value"))
        def update_step_delay(value: float):
            self.control_state.set_step_delay_seconds(value)
            return f"Delay: {value:.2f}s / step"

        @app.callback(Output("simulation-control-status", "children"), Input("start-simulation-button", "n_clicks"))
        def request_start(n_clicks: int):
            if n_clicks:
                self.control_state.request_simulation_start()
                return "Simulation running"
            return "Waiting for start"

        return app

    def _build_elements(self, snapshot: RendererSnapshot) -> list[dict]:
        elements: list[dict] = []
        for node in self.graph_spec.nodes:
            state = node.state
            classes = ["mdp-node"]
            if node.is_final:
                classes.append("terminal-node")
            if snapshot.visited_states.get(state, 0) > 0:
                classes.append("visited-node")
            if snapshot.current_state == state:
                classes.append("current-node")

            elements.append({
                "data": {
                    "id": self._state_id(state),
                    "label": state.name,
                    "visits": snapshot.visited_states.get(state, 0)
                },
                "position": NODE_POSITIONS[state],
                "classes": " ".join(classes),
                "locked": True
            })

        for edge in self.graph_spec.edges:
            edge_key = EdgeKey(source=edge.source, action=edge.action, target=edge.target)
            visits = snapshot.visited_edges.get(edge_key, 0)
            classes = ["mdp-edge"]
            if visits > 0:
                classes.append("visited-edge")
            if snapshot.current_edge == edge_key:
                classes.append("current-edge")
            elements.append({
                "data": {
                    "id": self._edge_id(edge),
                    "source": self._state_id(edge.source),
                    "target": self._state_id(edge.target),
                    "label": (f"{edge.action.name} | p={edge.probability: .2f} | r={edge.reward: .1f}"),
                    "visits": visits
                },
                "classes": " ".join(classes)
            })
        return elements

    def _build_status(self, snapshot: RendererSnapshot) -> list:
        current_state = snapshot.current_state.name if snapshot.current_state else "-"
        last_reward = "-" if snapshot.last_reward is None else f"{snapshot.last_reward:.2f}"
        return [
            html.Div(f"Episode: {snapshot.episode_index}"),
            html.Div(f"Step: {snapshot.step_index}"),
            html.Div(f"State: {current_state}"),
            html.Div(f"Reward: {last_reward}"),
            html.Div(f"Total reward: {snapshot.total_reward:.2f}"),
            html.Div(f"Terminated: {snapshot.terminated}")
        ]

    def _build_state_path(self, snapshot: RendererSnapshot):
        if not snapshot.state_path:
            return "-"
        return " -> ".join(self._display_name(state) for state in snapshot.state_path)

    def _build_outcome_counts(self, snapshot: RendererSnapshot) -> list:
        terminal_states = [State.DESTROYED, State.BASE, State.IMMOBILE]
        return [
            html.Div(
                [
                    html.Span(f"{self._display_name(state)}: {str(snapshot.outcome_counts.get(state,0))}")
                ],
                className="outcome-row"
            )
            for state in terminal_states
        ]

    def _build_policy_matrix(self):
        if self.policy_table is None:
            return html.Div("Policy unavailable")

        rows = [
            html.Tr(
                [html.Th("State")]
                + [html.Th(action.name) for action in Action]
            )
        ]
        for state in State:
            action_distribution = self.policy_table.get(state, {})
            rows.append(
                html.Tr(
                    [html.Td(self._display_name(state))]
                    + [
                        html.Td(f"{action_distribution.get(action, 0.0):.2f}")
                        for action in Action
                    ]
                )
            )
        return html.Table(rows)

    def _state_id(self, state: State) -> str:
        return state.name

    def _display_name(self, state: State) -> str:
        return state.name.replace("_", " ").title().replace(" ", "")

    def _edge_id(self, edge: GraphEdge) -> str:
        return f"{self._state_id(edge.source)}-{edge.action.name}-{self._state_id(edge.target)}"

    # TODO: move to config file / config loader
    def _stylesheet(self) -> list[dict]:
        return [
            {
                "selector": "node",
                "style": {
                    "content": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "13px",
                    "font-family": "Inter, Segoe UI, sans-serif",
                    "background-color": "#1f2937",
                    "border-width": 2,
                    "border-color": "#6b7280",
                    "color": "#e5e7eb",
                    "width": 74,
                    "height": 74
                }
            },
            {
                "selector": ".terminal-node",
                "style": {
                    "background-color": "#0f766e",
                    "border-color": "#5eead4"
                }
            },
            {
                "selector": ".current-node",
                "style": {
                    "background-color": "#facc15",
                    "border-color": "#fef3c7",
                    "color": "#111827",
                    "width": 90,
                    "height": 90
                }
            },
            {
                "selector": "edge",
                "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "line-color": "#4b5563",
                    "target-arrow-color": "#4b5563",
                    "width": 2,
                    "content": "data(label)",
                    "font-size": "10px",
                    "color": "#cbd5e1",
                    "text-background-color": "#111827",
                    "text-background-opacity": 0.85,
                    "text-background-padding": 3,
                    "text-rotation": "autorotate"
                }
            },
            {
                "selector": ".visited-edge",
                "style": {
                    "line-color": "#38bdf8",
                    "target-arrow-color": "#38bdf8",
                    "width": 4,
                    "opacity": 0.9
                }
            },
            {
                "selector": ".current-edge",
                "style": {
                    "line-color": "#facc15",
                    "target-arrow-color": "#facc15",
                    "width": 7,
                    "opacity": 1.0
                }
            }
        ]
