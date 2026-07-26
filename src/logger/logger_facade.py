from __future__ import annotations
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from logger.events import RunStarted, EpisodeStarted, StepLogged, TerminalReason, EpisodeFinished, RunFinished, \
    ErrorEvent
from logger.sinks import Sink
from model.sets.states import State
from model.transition import Transition


class LoggerFacade:
    def __init__(self, sink: Sink, run_id: str | None = None) -> None:
        self.sink = sink
        self.run_id = run_id or str(uuid4())

    def run_started(self, seed: int | None, num_episodes: int, max_steps: int, policy_path: str | Path) -> None:
        event = RunStarted(
            run_id=self.run_id,
            timestamp=datetime.now(),
            seed=seed,
            num_episodes=num_episodes,
            max_steps=max_steps,
            policy_path=str(policy_path)
        )
        self.sink.emit(event)

    def episode_started(self, episode_index: int) -> None:
        event = EpisodeStarted(
            run_id=self.run_id,
            timestamp=datetime.now(),
            episode_index=episode_index
        )
        self.sink.emit(event)

    def step_logged(self, episode_index: int, step_index: int, transition: Transition) -> None:
        event = StepLogged(
            run_id=self.run_id,
            timestamp=datetime.now(),
            episode_index=episode_index,
            step_index=step_index,
            transition=transition
        )
        self.sink.emit(event)

    def episode_finished(self, episode_index: int, total_reward: float, steps: int, terminated: bool, final_state: State) -> None:
        reason: TerminalReason = "terminal_state" if terminated else "max_steps"
        event = EpisodeFinished(
            run_id=self.run_id,
            timestamp=datetime.now(),
            episode_index=episode_index,
            total_reward=total_reward,
            steps=steps,
            terminated=terminated,
            final_state=final_state,
            reason=reason
        )
        self.sink.emit(event)

    def run_finished(self, episodes: int, terminal_episodes: int, truncated_episodes: int, average_reward: float, total_reward: float) -> None:
        event = RunFinished(
            run_id=self.run_id,
            timestamp=datetime.now(),
            episodes=episodes,
            terminal_episodes=terminal_episodes,
            truncated_episodes=truncated_episodes,
            average_reward=average_reward,
            total_reward=total_reward
        )
        self.sink.emit(event)

    def error(self, error: Exception) -> None:
        event = ErrorEvent(
            run_id=self.run_id,
            timestamp=datetime.now(),
            message=str(error),
            error_type=type(error).__name__
        )
        self.sink.emit(event)

    def close(self) -> None:
        self.sink.close()
