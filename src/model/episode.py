from dataclasses import dataclass
from model.transition import Transition
from model.distributions.policy import Policy
from model.environment import Environment


@dataclass
class EpisodeResult:
    transitions: list[Transition]
    total_reward: float
    terminated: bool

def run_episode(environment: Environment, policy: Policy, max_steps: int = 100) -> EpisodeResult:
    state = environment.reset()
    transitions: list[Transition] = []
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy.sample_action(state = state, rng = environment.rng)
        result = environment.step(action = action)
        transition = Transition(state = state, action = action, reward = result.reward, next_state = result.state, terminated = result.terminated)
        transitions.append(transition)
        total_reward += result.reward
        state = result.state
        if result.terminated:
            return EpisodeResult(transitions = transitions, total_reward = total_reward, terminated = True)
    return EpisodeResult(transitions = transitions, total_reward = total_reward, terminated = False)