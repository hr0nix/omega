import abc
import jax

from .agent import Agent


class TrainableAgentBase(Agent):
    @abc.abstractmethod
    def train_on_batch(self, trajectory_batch):
        pass

    @abc.abstractmethod
    def try_load_from_checkpoint(self, path):
        pass

    @abc.abstractmethod
    def save_to_checkpoint(self, path):
        pass


class JaxTrainableAgentBase(TrainableAgentBase):
    def __init__(self, *args, **kwargs):
        super(JaxTrainableAgentBase, self).__init__(*args, **kwargs)

        self._random_key = jax.random.PRNGKey(31337)

    def next_random_key(self):
        self._random_key, subkey = jax.random.split(self._random_key)
        return subkey
