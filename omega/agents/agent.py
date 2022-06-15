import abc


class Agent(abc.ABC):
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @abc.abstractmethod
    def act_on_batch(self, observation_batch, memory):
        pass

    def init_memory_batch(self, batch_size):
        return {}

    def update_memory_batch(self, prev_memory, new_memory_state, actions, done):
        return prev_memory
