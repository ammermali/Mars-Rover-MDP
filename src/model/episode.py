from dataclasses import dataclass
from model.transition import Transition
from model.distributions.policy import Policy
from model.environment import Environment
from model.episode_observer import EpisodeObserver


@dataclass
class EpisodeResult:
    transitions: list[Transition]
    total_reward: float
    terminated: bool

def run_episode(environment: Environment, policy: Policy, max_steps: int, observer: EpisodeObserver | None = None, episode_index: int = 0) -> EpisodeResult:
    state = environment.reset()
    transitions: list[Transition] = []
    total_reward = 0.0
    if observer is not None:
        observer.on_episode_started(episode_index)
    for step_index in range(max_steps):
        action = policy.sample_action(state = state, rng = environment.rng)
        result = environment.step(action = action)
        transition = Transition(state = state, action = action, reward = result.reward, next_state = result.state, terminated = result.terminated)
        transitions.append(transition)
        total_reward += result.reward
        state = result.state
        if observer is not None:
            observer.on_step(episode_index, step_index, transition)
        if result.terminated:
            if observer is not None:
                observer.on_episode_finished(episode_index=episode_index, total_reward=total_reward, steps=len(transitions), terminated=True, final_state=state)
            return EpisodeResult(transitions = transitions, total_reward = total_reward, terminated = True)
    if observer is not None:
        observer.on_episode_finished(episode_index=episode_index, total_reward=total_reward, steps=len(transitions), terminated=False, final_state=state)
    return EpisodeResult(transitions = transitions, total_reward = total_reward, terminated = False)
