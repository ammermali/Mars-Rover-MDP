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

class CompositeEpisodeObserver(EpisodeObserver):
    def __init__(self, observers: list[EpisodeObserver]) -> None:
        self.observers = observers

    def on_episode_started(self, episode_index: int) -> None:
        for observer in self.observers:
            observer.on_episode_started(episode_index)

    def on_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        for observer in self.observers:
            observer.on_step(episode_index, step_index, transition)

    def on_episode_finished(self, episode_index: int, total_reward: float, steps: int, terminated: bool, final_state: State) -> None:
        for observer in self.observers:
            observer.on_episode_finished(episode_index, total_reward, steps, terminated, final_state)