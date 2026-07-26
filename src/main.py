from config.config_loader import load_config
from model.environment import Environment
from model.distributions.policy import Policy
from model.episode_observer import EpisodeObserver, CompositeEpisodeObserver
from model.episode import EpisodeResult, run_episode
from logger.episode_observer import LoggingEpisodeObserver
from logger.logger_facade import LoggerFacade
from logger.sinks import CompositeSink, ConsoleSink, JsonFileSink
import random
from uuid import uuid4
from renderer.cytoscape_renderer import CytoscapeRenderer
from visualization.control_state import VisualizationControlState
from visualization.visualization_state import VisualizationState
from visualization.live_visualization_observer import LiveVisualizationObserver

def main() -> None:
    config = load_config()
    run_id = str(uuid4())
    log_path = config.logging.directory / f"{run_id}.json"
    sink = CompositeSink([
        ConsoleSink(include_steps=False),
        JsonFileSink(log_path)
    ])
    logger = LoggerFacade(sink=sink, run_id=run_id)
    renderer = None

    try:
        policy = Policy()
        policy.load_from_yaml(config.policy.path)
        visualization_control_state = VisualizationControlState(step_delay_seconds=config.visualization.step_delay_seconds)
        renderer = CytoscapeRenderer(
            policy_table=policy.get(),
            control_state=visualization_control_state,
            host=config.visualization.host,
            port=config.visualization.port,
            poll_interval_ms=config.visualization.poll_interval_ms
        )
        renderer.start()
        visualization_control_state.wait_for_simulation_start()
        logger.run_started(
            seed=config.experiment.seed,
            num_episodes=config.experiment.num_episodes,
            max_steps=config.experiment.max_steps,
            policy_path=config.policy.path
        )
        visualization_observer = LiveVisualizationObserver(
            state = VisualizationState(),
            renderer = renderer,
            control_state = visualization_control_state
        )
        observer = CompositeEpisodeObserver([
            LoggingEpisodeObserver(logger=logger),
            visualization_observer
        ])
        rng = random.Random(config.experiment.seed)

        total_reward = 0.0
        terminal_episodes = 0
        truncated_episodes = 0

        for episode_index in range(config.experiment.num_episodes):
            result = episode_runner(
                episode_index=episode_index,
                rng=rng,
                policy=policy,
                observer=observer,
                max_steps=config.experiment.max_steps
            )
            total_reward += result.total_reward
            if result.terminated:
                terminal_episodes += 1
            else:
                truncated_episodes += 1

        average_reward = total_reward / config.experiment.num_episodes if config.experiment.num_episodes else 0.0
        logger.run_finished(
            episodes=config.experiment.num_episodes,
            terminal_episodes=terminal_episodes,
            truncated_episodes=truncated_episodes,
            average_reward=average_reward,
            total_reward=total_reward
        )
    except Exception as error:
        logger.error(error)
        raise
    finally:
        logger.close()
        if renderer is not None:
            renderer.close()


def episode_runner(episode_index: int, rng: random.Random, policy: Policy, observer: EpisodeObserver, max_steps: int) -> EpisodeResult:
    environment = Environment(rng)
    return run_episode(environment=environment, policy=policy, max_steps=max_steps, observer=observer, episode_index=episode_index)

if __name__ == "__main__":
    main()
