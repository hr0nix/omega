import attr
from collections import defaultdict


@attr.s
class PerRunStats(object):
    steps = attr.ib(type=int, default=0)
    total_reward = attr.ib(type=float, default=0.0)


class EvaluationStats(object):
    def __init__(self):
        self._per_run_stats = None

        self.reset()

    def reset(self):
        self._per_run_stats = defaultdict(PerRunStats)

    def add_stats(self, run_index, reward):
        s = self._per_run_stats[run_index]
        s.steps += 1
        s.total_reward += reward

    def to_dict(self):
        num_episodes = len(self._per_run_stats)
        num_steps = sum(rs.steps for rs in self._per_run_stats.values())
        reward_sum = sum(rs.total_reward for rs in self._per_run_stats.values())

        return {
            'timestamps_per_episode': float(num_steps) / num_episodes,
            'reward_per_episode': reward_sum / num_episodes,
            'reward_per_step': reward_sum / num_steps,
        }

    def print_summary(self, title=None):
        stats = self.to_dict()
        title = title or 'Evaluation summary:'

        print(title, flush=True)
        print('  Timestamps per episode: {}'.format(stats['timestamps_per_episode']), flush=True)
        print('  Per-episode avg reward: {}'.format(stats['reward_per_episode']), flush=True)
        print('  Per-step avg reward: {}'.format(stats['reward_per_step']), flush=True)
