import attr
import numpy as np


@attr.s
class Transition(object):
    """
    The observation of a state we're currently in.
    """
    observation = attr.ib()
    """
    The action that took us here.
    """
    action = attr.ib()
    """
    The reward for getting in this state. 
    """
    reward = attr.ib(type=float)
    """
    Whether this is a terminal state.
    """
    done = attr.ib(type=bool)
    """
    Metadata associated with the transition to this state (action log probs, state values etc.)
    """
    metadata = attr.ib(type=dict)


class Trajectory(object):
    def __init__(self, initial_state):
        self._initial_state = initial_state
        self._transitions = []

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def transitions(self):
        return self._transitions

    def add_transition(self, action, observation, reward, done, metadata):
        self._transitions.append(
            Transition(
                action=action, observation=observation, reward=reward, done=done, metadata=metadata
            )
        )


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
