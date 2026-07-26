import json
import sys
from pathlib import Path
from typing import Protocol, TextIO
from logger.events import Event, RunFinished, ErrorEvent, RunStarted, EpisodeStarted, StepLogged, EpisodeFinished
from logger.serializer import serialize_event, JsonDict


class Sink(Protocol):
    def emit(self, event: Event) -> None:
        ...
    def close(self) -> None:
        ...

class ConsoleSink(Sink):
    def __init__(self, include_steps: bool = False, stream: TextIO | None = None):
        self.include_steps = include_steps
        self.stream = stream if stream is not None else sys.stdout

    def emit(self, event: Event) -> None:
        message = self._format_event(event)
        if message is not None:
            print(message, file=self.stream)

    def close(self) -> None:
        pass

    def _format_event(self, event: Event) -> str | None:
        if isinstance(event, RunStarted):
            return f"Run {event.run_id} started."
        if isinstance(event, EpisodeStarted):
            return f"Episode {event.episode_index} started."
        if isinstance(event, StepLogged):
            if not self.include_steps:
                return None
            transition = event.transition
            return (
                f"Episode {event.episode_index}, step {event.step_index}: "
                f"{transition.state.name} ---{transition.action.name} / {transition.reward: .2f} --> {transition.next_state.name} "
            )

        if isinstance(event, EpisodeFinished):
            return f"Episode {event.episode_index} finished. Total reward: {event.total_reward: .2f}"

        if isinstance(event, RunFinished):
            return (
                f"Run {event.run_id} finished: episodes={event.episodes}, "
                f"terminal={event.terminal_episodes}, truncated={event.truncated_episodes}, "
                f"average_reward={event.average_reward:.2f}, total_reward={event.total_reward:.2f}"
            )

        if isinstance(event, ErrorEvent):
            return f"Error: {event.error_type}: {event.message}"

        return None

class JsonFileSink(Sink):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.events: list[JsonDict] = []

    def emit(self, event: Event) -> None:
        self.events.append(serialize_event(event))

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"run_id": self._run_id(), "events": self.events}
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def close(self) -> None:
        self.flush()

    def _run_id(self) -> str | None:
        if not self.events:
            return None
        return self.events[0]["run_id"]


class CompositeSink(Sink):
    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
