from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from model.sets.actions import Action
from model.sets.states import State


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "mdp.yaml"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    num_episodes: int
    max_steps: int


@dataclass(frozen=True)
class PolicyConfig:
    path: Path


@dataclass(frozen=True)
class LoggingConfig:
    directory: Path


@dataclass(frozen=True)
class StepDelaySliderConfig:
    min_seconds: float
    max_seconds: float
    step_seconds: float
    marks: dict[float, str]


@dataclass(frozen=True)
class VisualizationConfig:
    step_delay_seconds: float
    host: str
    port: int
    poll_interval_ms: int
    step_delay_slider: StepDelaySliderConfig
    node_positions: dict[State, dict[str, float]]


@dataclass(frozen=True)
class MdpConfig:
    transitions: dict[tuple[State, Action], dict[State, float]]
    rewards: dict[tuple[State, Action, State], float]


@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig
    policy: PolicyConfig
    logging: LoggingConfig
    visualization: VisualizationConfig
    mdp: MdpConfig


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, Mapping):
        raise ValueError("Configuration file must contain a mapping.")

    return AppConfig(
        experiment=_load_experiment_config(raw_config["experiment"]),
        policy=_load_policy_config(raw_config["policy"]),
        logging=_load_logging_config(raw_config["logging"]),
        visualization=_load_visualization_config(raw_config["visualization"]),
        mdp=_load_mdp_config(raw_config["mdp"]),
    )


def load_mdp_config(path: str | Path | None = None) -> MdpConfig:
    return load_config(path).mdp


def _load_experiment_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        seed=int(raw["seed"]),
        num_episodes=int(raw["num_episodes"]),
        max_steps=int(raw["max_steps"]),
    )


def _load_policy_config(raw: Mapping[str, Any]) -> PolicyConfig:
    return PolicyConfig(path=_project_path(raw["path"]))


def _load_logging_config(raw: Mapping[str, Any]) -> LoggingConfig:
    return LoggingConfig(directory=_project_path(raw["directory"]))


def _load_visualization_config(raw: Mapping[str, Any]) -> VisualizationConfig:
    graph = raw["graph"]
    layout = graph["layout"]
    return VisualizationConfig(
        step_delay_seconds=float(raw["step_delay_seconds"]),
        host=str(raw["host"]),
        port=int(raw["port"]),
        poll_interval_ms=int(raw["poll_interval_ms"]),
        step_delay_slider=_load_step_delay_slider_config(raw["step_delay_slider"]),
        node_positions={
            _state_from_name(state_name): {
                "x": float(position["x"]),
                "y": float(position["y"]),
            }
            for state_name, position in layout.items()
        },
    )


def _load_step_delay_slider_config(raw: Mapping[str, Any]) -> StepDelaySliderConfig:
    return StepDelaySliderConfig(
        min_seconds=float(raw["min_seconds"]),
        max_seconds=float(raw["max_seconds"]),
        step_seconds=float(raw["step_seconds"]),
        marks={float(value): str(label) for value, label in raw["marks"].items()},
    )


def _load_mdp_config(raw: Mapping[str, Any]) -> MdpConfig:
    return MdpConfig(
        transitions=_load_transitions(raw["transitions"]),
        rewards=_load_rewards(raw["rewards"]),
    )


def _load_transitions(raw_transitions: list[Mapping[str, Any]]) -> dict[tuple[State, Action], dict[State, float]]:
    transitions: dict[tuple[State, Action], dict[State, float]] = {}
    for transition in raw_transitions:
        state = _state_from_name(transition["state"])
        action = _action_from_name(transition["action"])
        next_states = {
            _state_from_name(next_state_name): float(probability)
            for next_state_name, probability in transition["next_states"].items()
        }
        transitions[(state, action)] = next_states
    return transitions


def _load_rewards(raw_rewards: list[Mapping[str, Any]]) -> dict[tuple[State, Action, State], float]:
    rewards: dict[tuple[State, Action, State], float] = {}
    for reward in raw_rewards:
        state = _state_from_name(reward["state"])
        action = _action_from_name(reward["action"])
        next_state = _state_from_name(reward["next_state"])
        rewards[(state, action, next_state)] = float(reward["value"])
    return rewards


def _state_from_name(name: str) -> State:
    try:
        return State[name]
    except KeyError as error:
        raise ValueError(f"Unknown state: {name}") from error


def _action_from_name(name: str) -> Action:
    try:
        return Action[name]
    except KeyError as error:
        raise ValueError(f"Unknown action: {name}") from error


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value
