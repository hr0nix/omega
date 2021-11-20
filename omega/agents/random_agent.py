import numpy as np
from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
    
    def act_on_batch(self, observation_batch):
        tensor_keys = list(observation_batch.keys())
        batch_size = observation_batch[tensor_keys[0]].shape[0]
        assert all(observation_batch[key].shape[0] == batch_size for key in tensor_keys)

        return np.random.randint(
            low=0,
            high=self.action_space.n - 1,
            size=(batch_size, )
        )

    def train_on_batch(self, trajectory_batch):
        pass
