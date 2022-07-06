import numpy as np
import jax.numpy as jnp

from omega.utils import pytree

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Optional
from functools import reduce


@dataclass
class RunningEpisodeStats(object):
    rewards: List[float] = field(default_factory=list)
    actions: List = field(default_factory=list)


@dataclass
class FinishedEpisodeStats(object):
    reward_sum: Optional[float] = None
    discounted_reward_sum: Optional[float] = None
    num_steps: Optional[int] = None


class ExponentialSmoother(object):
    def __init__(self, smoothing):
        self._value = 0.0
        self._smoothing = smoothing

    def add(self, value):
        self._value = self._smoothing * self._value + (1.0 - self._smoothing) * value

    @property
    def smoothed_value(self):
        return self._value


class EvaluationStats(object):
    AVERAGE_REWARD_OVER_NUM_EPISODES = 100

    def __init__(self, discount_factor, rolling_stats_smoothing=0.9):
        self._rolling_stats = defaultdict(lambda: ExponentialSmoother(smoothing=rolling_stats_smoothing))
        self._discount_factor = discount_factor
        self._running_episode_stats = defaultdict(RunningEpisodeStats)
        self._finished_episode_stats = list()
        self._total_steps = 0
        self._num_finished_episodes = 0
        self._unique_rewards = set()

    def add_rolling_stats(self, values_dict):
        for key, value in values_dict.items():
            self._rolling_stats[key].add(value)

    def add_transition(self, episode_index, action, reward, done):
        stats = self._running_episode_stats[episode_index]
        stats.actions.append(action)
        stats.rewards.append(reward)

        self._unique_rewards.add(reward)

        if done:
            final_stats = self._finalize_episode_stats(stats)
            self._total_steps += final_stats.num_steps
            self._num_finished_episodes += 1

            del self._running_episode_stats[episode_index]
            self._finished_episode_stats.append(final_stats)
            # Do not remember too many stats
            if len(self._finished_episode_stats) > self.AVERAGE_REWARD_OVER_NUM_EPISODES:
                self._finished_episode_stats = self._finished_episode_stats[-self.AVERAGE_REWARD_OVER_NUM_EPISODES:]

    def _finalize_episode_stats(self, stats):
        final_stats = FinishedEpisodeStats()
        final_stats.num_steps = len(stats.rewards)
        final_stats.reward_sum = sum(stats.rewards)
        final_stats.discounted_reward_sum = reduce(
            lambda accum, r: r + self._discount_factor * accum, stats.rewards[::-1], 0.0)
        return final_stats

    def to_dict(self, include_rolling_stats=False):
        if len(self._finished_episode_stats) == 0:
            return dict()

        result = {
            'total_steps': self._total_steps,
            'total_finished_episodes': self._num_finished_episodes,
            'last_episode_total_reward': self._finished_episode_stats[-1].reward_sum,
            'last_episode_steps': self._finished_episode_stats[-1].num_steps,
            'last_100_episode_avg_reward': np.mean(
                [s.reward_sum for s in self._finished_episode_stats[-self.AVERAGE_REWARD_OVER_NUM_EPISODES:]]
            ),
            'last_100_episode_avg_discounted_reward': np.mean(
                [s.discounted_reward_sum for s in self._finished_episode_stats[-self.AVERAGE_REWARD_OVER_NUM_EPISODES:]]
            ),
        }
        if include_rolling_stats:
            result.update({
                key: value.smoothed_value
                for key, value in self._rolling_stats.items()
            })
        return result

    def print_summary(self, title=None):
        if len(self._finished_episode_stats) == 0:
            print('No finished episodes recorded yet.')
            return

        stats = self.to_dict(include_rolling_stats=False)
        stats = pytree.update(stats, {
            'unique_rewards': self._unique_rewards,
        })
        title = title or 'Evaluation summary:'

        print(title, flush=True)
        for key_title, key in [
            ('Total steps', 'total_steps'),
            ('Total finished episodes', 'total_finished_episodes'),
            ('Last episode total reward', 'last_episode_total_reward'),
            ('Last episode steps', 'last_episode_steps'),
            ('Last 100 episodes avg reward', 'last_100_episode_avg_reward'),
            ('Last 100 episodes avg discounted reward', 'last_100_episode_avg_discounted_reward'),
            ('Unique rewards', 'unique_rewards'),
        ]:
            print('  {}: {}'.format(key_title, stats[key]), flush=True)
