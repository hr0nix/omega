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

    def print_summary(self, title=None):
        num_episodes = len(self._per_run_stats)
        num_steps = sum(rs.steps for rs in self._per_run_stats.values())
        reward_sum = sum(rs.steps for rs in self._per_run_stats.values())

        title = title or 'Evaluation summary:'
        print(title)
        print('  Episodes: {}'.format(num_episodes))
        print('  Per-episode avg reward: {}'.format(reward_sum / num_episodes))
        print('  Per-step avg reward: {}'.format(reward_sum / num_steps))
