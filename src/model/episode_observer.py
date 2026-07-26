from typing import Protocol

from model.sets.states import State
from model.transition import Transition


class EpisodeObserver(Protocol):
    def on_episode_started(self, episode_index: int) -> None:
        ...

    def on_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        ...

    def on_episode_finished(self, episode_index: int, total_reward: float, steps: int, terminated: bool, final_state: State) -> None:
        ...

