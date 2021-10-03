import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List, Optional


@dataclass
class RunningEpisodeStats(object):
    rewards: List[float] = field(default_factory=list)
    actions: List = field(default_factory=list)


@dataclass
class FinishedEpisodeStats(object):
    reward_sum: Optional[float] = None
    min_reward: Optional[float] = None
    max_reward: Optional[float] = None
    reward_p50: Optional[float] = None
    reward_p75: Optional[float] = None
    reward_p95: Optional[float] = None
    reward_p99: Optional[float] = None
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
    def __init__(self, rolling_stats_smoothing=0.9):
        self._rolling_stats = defaultdict(lambda: ExponentialSmoother(smoothing=rolling_stats_smoothing))
        self._running_episode_stats = defaultdict(RunningEpisodeStats)
        self._finished_episodes = set()
        self._finished_episode_stats = list()
        self._total_steps = 0

    def add_rolling_stats(self, values_dict):
        for key, value in values_dict.items():
            self._rolling_stats[key].add(value)

    def add_transition(self, episode_index, action, reward, done):
        if episode_index is self._finished_episodes:
            raise ValueError('Trying to add transition to a finished episode {}'.format(episode_index))

        # To handle JAX/numpy arrays which are not hashable
        if isinstance(action, (jnp.ndarray, np.ndarray)):
            if len(action.shape) > 0:
                raise ValueError('Only scalar actions are currently supported')
            action = np.asscalar(action)

        stats = self._running_episode_stats[episode_index]
        stats.actions.append(action)
        stats.rewards.append(reward)

        if done:
            self._finished_episodes.add(episode_index)
            final_stats = self._finalize_episode_stats(stats)
            del self._running_episode_stats[episode_index]
            self._total_steps += final_stats.num_steps

    def _finalize_episode_stats(self, stats):
        sorted_rewards = list(stats.rewards)
        sorted_rewards.sort()
        action_counter = Counter(stats.actions)

        final_stats = FinishedEpisodeStats()
        final_stats.num_steps = len(sorted_rewards)
        final_stats.reward_sum = sum(sorted_rewards)
        final_stats.min_reward = min(sorted_rewards)
        final_stats.max_reward = max(sorted_rewards)
        final_stats.reward_p50 = sorted_rewards[int(len(sorted_rewards) * 0.50)]
        final_stats.reward_p75 = sorted_rewards[int(len(sorted_rewards) * 0.75)]
        final_stats.reward_p95 = sorted_rewards[int(len(sorted_rewards) * 0.95)]
        final_stats.reward_p99 = sorted_rewards[int(len(sorted_rewards) * 0.99)]
        final_stats.top_5_actions = action_counter.most_common(5)

        self._finished_episode_stats.append(final_stats)

        return final_stats

    def to_dict(self, include_non_scalar_stats=False, include_rolling_stats=False):
        if len(self._finished_episode_stats) == 0:
            return dict()

        result = {
            'total_steps': self._total_steps,
            'total_finished_episodes': len(self._finished_episode_stats),
            'last_episode_total_reward': self._finished_episode_stats[-1].reward_sum,
            'last_episode_steps': self._finished_episode_stats[-1].num_steps,
            'last_episode_min_reward': self._finished_episode_stats[-1].min_reward,
            'last_episode_max_reward': self._finished_episode_stats[-1].max_reward,
            'last_episode_reward_p50': self._finished_episode_stats[-1].reward_p50,
            'last_episode_reward_p75': self._finished_episode_stats[-1].reward_p75,
            'last_episode_reward_p95': self._finished_episode_stats[-1].reward_p95,
            'last_episode_reward_p99': self._finished_episode_stats[-1].reward_p99,
            'last_100_episode_avg_reward': np.mean([s.reward_sum for s in self._finished_episode_stats[-100:]])
        }
        if include_non_scalar_stats:
            result.update({
                'last_episode_top_5_actions': self._finished_episode_stats[-1].top_5_actions
            })
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

        stats = self.to_dict(include_non_scalar_stats=True, include_rolling_stats=False)
        title = title or 'Evaluation summary:'

        print(title, flush=True)
        for key_title, key in [
            ('Total steps', 'total_steps'),
            ('Total finished episodes', 'total_finished_episodes'),
            ('Last episode total reward', 'last_episode_total_reward'),
            ('Last episode steps', 'last_episode_steps'),
            ('Last episode min reward', 'last_episode_min_reward'),
            ('Last episode max reward', 'last_episode_max_reward'),
            ('Last episode reward 50%', 'last_episode_reward_p50'),
            ('Last episode reward 75%', 'last_episode_reward_p75'),
            ('Last episode reward 95%', 'last_episode_reward_p95'),
            ('Last episode reward 99%', 'last_episode_reward_p99'),
            ('Last episode top-5 actions', 'last_episode_top_5_actions'),
            ('Last 100 episodes avg reward', 'last_100_episode_avg_reward'),
        ]:
            print('  {}: {}'.format(key_title, stats[key]), flush=True)

