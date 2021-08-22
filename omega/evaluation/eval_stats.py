import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import List


@dataclass
class TrajectoryStats(object):
    rewards: List[float] = field(default_factory=list)
    actions: List = field(default_factory=list)
    done: bool = False


class EvaluationStats(object):
    def __init__(self):
        self._per_trajectory_stats = None
        self.reset()

    def reset(self):
        self._per_trajectory_stats = defaultdict(TrajectoryStats)

    def add_transition(self, trajectory_index, action, reward, done):
        # To handle JAX/numpy arrays which are not hashable
        if isinstance(action, (jnp.ndarray, np.ndarray)):
            if len(action.shape) > 0:
                raise ValueError('Only scalar actions are currently supported')
            action = np.asscalar(action)

        s = self._per_trajectory_stats[trajectory_index]
        assert not s.done
        s.actions.append(action)
        s.rewards.append(reward)
        s.done = done

    def to_dict(self, include_non_scalar_stats=False):
        num_finished_episodes = sum(ts.done for ts in self._per_trajectory_stats.values())

        all_rewards = sum((ts.rewards for ts in self._per_trajectory_stats.values()), [])
        all_rewards.sort()
        reward_sum = sum(all_rewards)
        max_reward = max(all_rewards)
        reward_p50 = all_rewards[int(len(all_rewards) * 0.5)]
        reward_p95 = all_rewards[int(len(all_rewards) * 0.95)]
        reward_p99 = all_rewards[int(len(all_rewards) * 0.99)]

        all_actions = sum((ts.actions for ts in self._per_trajectory_stats.values()), [])
        action_counter = Counter(all_actions)

        num_steps = len(all_rewards)

        result = {
            'max_reward': max_reward,
            'reward_p50': reward_p50,
            'reward_p95': reward_p95,
            'reward_p99': reward_p99,
            'reward_per_step': reward_sum / num_steps,
            'done_per_step': float(num_finished_episodes) / num_steps,
        }
        if include_non_scalar_stats:
            result.update({
                'top_5_actions': action_counter.most_common(5)
            })
        return result

    def print_summary(self, title=None):
        stats = self.to_dict(include_non_scalar_stats=True)
        title = title or 'Evaluation summary:'

        print(title, flush=True)
        for key_title, key in [
            ('Reward per step', 'reward_per_step'),
            ('Max reward', 'max_reward'),
            ('Reward 50%', 'reward_p50'),
            ('Reward 95%', 'reward_p95'),
            ('Reward 99%', 'reward_p99'),
            ('Finished episodes per step', 'done_per_step'),
            ('Top-5 actions', 'top_5_actions'),
        ]:
            print('  {}: {}'.format(key_title, stats[key]), flush=True)

