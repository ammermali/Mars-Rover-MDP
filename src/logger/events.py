from dataclasses import dataclass
from typing import Literal
from datetime import datetime
from model.sets.states import State

from model.transition import Transition

EventType = Literal["run_started", "episode_started", "step_logged", "episode_finished", "run_finished", "error"]
TerminalReason = Literal["max_steps", "terminal_state"]

@dataclass(frozen=True)
class Event:
    run_id: str
    timestamp: datetime
    @property
    def event_type(self) -> EventType:
        raise NotImplementedError

@dataclass(frozen=True)
class RunStarted(Event):
    seed: int | None
    num_episodes: int
    max_steps: int
    policy_path: str
    @property
    def event_type(self) -> EventType:
        return "run_started"

@dataclass(frozen=True)
class EpisodeStarted(Event):
    episode_index: int
    @property
    def event_type(self) -> EventType:
        return "episode_started"

@dataclass(frozen=True)
class StepLogged(Event):
    episode_index: int
    step_index: int
    transition: Transition
    @property
    def event_type(self) -> EventType:
        return "step_logged"

@dataclass(frozen=True)
class EpisodeFinished(Event):
    episode_index: int
    total_reward: float
    steps: int
    terminated: bool
    final_state: State
    reason: TerminalReason
    @property
    def event_type(self) -> EventType:
        return "episode_finished"

@dataclass(frozen=True)
class RunFinished(Event):
    episodes: int
    terminal_episodes: int
    truncated_episodes: int
    average_reward: float
    total_reward: float
    @property
    def event_type(self) -> EventType:
        return "run_finished"

@dataclass(frozen=True)
class ErrorEvent(Event):
    message: str
    error_type: str
    @property
    def event_type(self) -> EventType:
        return "error"
