from pathlib import Path
from model.environment import Environment
from model.distributions.policy import Policy
from model.episode_observer import EpisodeObserver
from model.episode import EpisodeResult, run_episode
from logger.episode_observer import LoggingEpisodeObserver
from logger.logger_facade import LoggerFacade
from logger.sinks import CompositeSink, ConsoleSink, JsonFileSink
import random
from uuid import uuid4

DEFAULT_POLICY_PATH = Path(__file__).parent.parent / "config" / "initial_policy.yaml"
LOGS_DIR = Path(__file__).parent.parent / "logs"
SEED = 42
NUM_EPISODES = 1000
MAX_STEPS = 100

def main() -> None:
    run_id = str(uuid4())
    log_path = LOGS_DIR / f"{run_id}.json"
    sink = CompositeSink([
        ConsoleSink(include_steps=False),
        JsonFileSink(log_path)
    ])
    logger = LoggerFacade(sink=sink, run_id=run_id)
    observer = LoggingEpisodeObserver(logger)

    try:
        logger.run_started(seed=SEED, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, policy_path=DEFAULT_POLICY_PATH)

        policy = Policy()
        policy.load_from_yaml(DEFAULT_POLICY_PATH)
        rng = random.Random(SEED)

        total_reward = 0.0
        terminal_episodes = 0
        truncated_episodes = 0

        for episode_index in range(NUM_EPISODES):
            result = episode_runner(episode_index=episode_index, rng=rng, policy=policy, observer=observer, max_steps=MAX_STEPS)
            total_reward += result.total_reward
            if result.terminated:
                terminal_episodes += 1
            else:
                truncated_episodes += 1

        average_reward = total_reward / NUM_EPISODES if NUM_EPISODES else 0.0
        logger.run_finished(episodes=NUM_EPISODES, terminal_episodes=terminal_episodes, truncated_episodes=truncated_episodes, average_reward=average_reward, total_reward=total_reward)
    except Exception as error:
        logger.error(error)
        raise
    finally:
        logger.close()


def episode_runner(episode_index: int, rng: random.Random, policy: Policy, observer: EpisodeObserver, max_steps: int) -> EpisodeResult:
    environment = Environment(rng)
    return run_episode(environment=environment, policy=policy, max_steps=max_steps, observer=observer, episode_index=episode_index)

if __name__ == "__main__":
    main()
