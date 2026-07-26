from dataclasses import dataclass

from logger.logger_facade import LoggerFacade
from model.episode_observer import EpisodeObserver
from model.sets.states import State
from model.transition import Transition


@dataclass(frozen=True)
class LoggingEpisodeObserver(EpisodeObserver):
    logger: LoggerFacade

    def on_episode_started(self, episode_index: int) -> None:
        self.logger.episode_started(episode_index)

    def on_step(self, episode_index: int, step_index: int, transition: Transition) -> None:
        self.logger.step_logged(episode_index=episode_index, step_index=step_index, transition=transition)

    def on_episode_finished(self, episode_index: int, total_reward: float, steps: int, terminated: bool, final_state: State) -> None:
        self.logger.episode_finished(episode_index=episode_index, total_reward=total_reward, steps=steps, terminated=terminated, final_state=final_state)
