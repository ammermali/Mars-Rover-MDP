from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from logger.events import ErrorEvent, EpisodeFinished, EpisodeStarted, Event, RunFinished, RunStarted, StepLogged
from model.transition import Transition

JsonDict = dict[str, Any]

def serialize_time(value: datetime) -> str:
    return value.isoformat()

def serialize_enum(value: Enum) -> str:
    return value.name

def serialize_path(value: Path | str) -> str:
    return str(value)

def serialize_transition(value: Transition) -> JsonDict:
    return {
        "state": serialize_enum(value.state),
        "action": serialize_enum(value.action),
        "reward": value.reward,
        "next_state": serialize_enum(value.next_state),
        "terminated": value.terminated
    }

def serialize_event(value: Event) -> JsonDict:
    base = {
        "type": value.event_type,
        "run_id": value.run_id,
        "timestamp": serialize_time(value.timestamp)
    }

    if isinstance(value, RunStarted):
        return {
            **base,
            "seed": value.seed,
            "num_episodes": value.num_episodes,
            "max_steps": value.max_steps,
            "policy_path": serialize_path(value.policy_path)
        }

    if isinstance(value, EpisodeStarted):
        return {
            **base,
            "episode_index": value.episode_index
        }

    if isinstance(value, StepLogged):
        return {
            **base,
            "episode_index": value.episode_index,
            "step_index": value.step_index,
            "transition": serialize_transition(value.transition)
        }

    if isinstance(value, EpisodeFinished):
        return {
            **base,
            "episode_index": value.episode_index,
            "total_reward": value.total_reward,
            "steps": value.steps,
            "terminated": value.terminated,
            "final_state": serialize_enum(value.final_state),
            "reason": value.reason
        }

    if isinstance(value, RunFinished):
        return {
            **base,
            "episodes": value.episodes,
            "terminal_episodes": value.terminal_episodes,
            "truncated_episodes": value.truncated_episodes,
            "average_reward": value.average_reward,
            "total_reward": value.total_reward
        }

    if isinstance(value, ErrorEvent):
        return {
            **base,
            "message": value.message,
            "error_type": value.error_type
        }

    raise TypeError(f"Unknown event type: {type(value)}")


def serialize_events(events: Iterable[Event]) -> list[JsonDict]:
    return [serialize_event(event) for event in events]
