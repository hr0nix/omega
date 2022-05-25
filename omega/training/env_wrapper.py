import gym


class EnvWrapper(gym.Env):
    def __init__(self, env):
        self._env = env

        self._current_state = None
        self._last_reward = None
        self._last_action = None
        self._is_done = None

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def current_state(self):
        return self._current_state

    @property
    def last_reward(self):
        return self._last_reward

    @property
    def last_action(self):
        return self._last_action

    @property
    def is_done(self):
        return self._is_done

    @property
    def is_running(self):
        return self._current_state is not None

    def reset(self):
        self._current_state = self._env.reset()
        self._is_done = False
        return self._current_state

    def step(self, action):
        assert self.is_running
        assert not self.is_done

        observation, reward, done, info = self._env.step(action)
        self._current_state = observation
        self._last_reward = reward
        self._last_action = action
        self._is_done = done
        return observation, reward, done, info

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.seed(seed)