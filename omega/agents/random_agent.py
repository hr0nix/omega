import numpy as np
from .agent import Agent

from ..utils import pytree


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
    
    def act_on_batch(self, observation_batch):
        batch_size = pytree.get_axis_dim(observation_batch, axis=0)
        random_actions = np.random.randint(low=0, high=self.action_space.n, size=(batch_size,))
        return random_actions, {}

    def train_on_batch(self, trajectory_batch):
        pass
