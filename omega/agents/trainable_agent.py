import abc

from .agent import Agent


class TrainableAgentBase(Agent):
    @abc.abstractmethod
    def train_on_batch(self, trajectory_batch):
        pass

    @abc.abstractmethod
    def try_load_from_checkpoint(self, path):
        pass

    @abc.abstractmethod
    def save_to_checkpoint(self, path, day):
        pass
