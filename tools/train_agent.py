import gym
import nle

from omega.agents import RandomAgent, NethackTransformerAgent
from omega.training import OnPolicyTrainer
from omega.evaluation import EvaluationStats


def env_factory():
    return gym.make("NetHackScore-v0")


def main():
    env = env_factory()
    agent = NethackTransformerAgent(env.observation_space, env.action_space)
    trainer = OnPolicyTrainer(
        batch_size=64,
        env_factory=env_factory,
        num_parallel_envs=4,
        num_day_steps=10,
        num_night_steps=1,
    )
    num_days = 10
    summarize_every_num_days = 2

    stats = EvaluationStats()
    for day in range(num_days):
        print('Day {}'.format(day))

        trainer.run_training_step(agent, stats)

        if (day + 1) % summarize_every_num_days == 0:
            stats.print_summary(title='After {} days:'.format(day + 1))
            stats.reset()


if __name__ == '__main__':
    main()
