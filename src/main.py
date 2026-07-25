from pathlib import Path
from model.environment import Environment
from model.distributions.policy import Policy
from model.episode import run_episode
import random

DEFAULT_POLICY_PATH = Path(__file__).parent.parent / "config" / "initial_policy.yaml"

def main() -> None:
    policy = Policy()
    policy.load_from_yaml(DEFAULT_POLICY_PATH)
    rng = random.Random(42)
    for i in range(1000):
        print(f"Starting simulation number {i}")
        episode_runner(rng, policy)


def episode_runner(rng: random.Random, policy: Policy) -> None:
    max_steps = 100
    environment = Environment(rng)
    result = run_episode(environment = environment, policy = policy, max_steps = max_steps)
    print(f"Terminated {result.terminated} after {result.total_reward:.2f} reward.")
    print(f"Steps {len(result.transitions)}")
    for transition in result.transitions:
        print(f"{transition.state.name} --{transition.action.name} / {transition.reward:.2f}--> {transition.next_state.name}")

if __name__ == "__main__":
    main()