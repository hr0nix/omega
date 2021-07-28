import attr
import numpy as np


@attr.s
class TrajectoryElement(object):
    action = attr.ib()
    observation = attr.ib()
    reward = attr.ib(type=float)
    done = attr.ib(type=bool)


class Trajectory(object):
    def __init__(self, initial_state):
        self._initial_state = initial_state
        self._elements = []

    @property
    def initial_state(self):
        return self._initial_state

    def __len__(self):
        return len(self._elements)

    @property
    def elements(self):
        return self._elements

    def compute_rewards_to_go(self):
        # TODO: support discount factor?
        result = np.zeros(shape=(len(self.elements),), dtype=np.float32)
        for i in range(len(self.elements)):
            prev_reward_to_go = result[-1 - i + 1] if i > 0 else 0.0
            result[-1 - i] = prev_reward_to_go + self._elements[-1 - i].reward
        return result

    def append(self, action, observation, reward, done):
        self._elements.append(TrajectoryElement(action, observation, reward, done))


class TrajectoryBatch(object):
    def __init__(self, trajectories=None):
        self._trajectories = trajectories or []

    @property
    def trajectories(self):
        return self._trajectories

    def add_trajectory(self, initial_state):
        trajectory = Trajectory(initial_state)
        self._trajectories.append(trajectory)
        return trajectory

    def sample_subbatch(self, batch_size):
        return TrajectoryBatch(np.random.choice(self._trajectories, size=batch_size).tolist())
