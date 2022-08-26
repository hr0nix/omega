import os.path

from minihack import MiniHackNavigation, RewardManager
from minihack.envs import register


DES_PATH = os.path.join(os.path.dirname(__file__), 'des')


class Retreat(MiniHackNavigation):
    """
    In this environment the agent can fight the monster to get a larger reward,
    but if things don't go well, the agent can retreat to the bottom-right corner.
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        reward_manager = RewardManager()
        reward_manager.add_coordinate_event(
            coordinates=(5, 6),
            reward=1.0,
            terminal_sufficient=True,
            terminal_required=True,
        )
        reward_manager.add_kill_event(
            name='killer bee',
            reward=2.0,
            terminal_sufficient=True,
            terminal_required=True,
        )
        super().__init__(
            *args,
            des_file=os.path.join(DES_PATH, 'retreat.des'),
            reward_manager=reward_manager,
            **kwargs
        )


register(
    id="MiniHack-Retreat-v0",
    entry_point="omega.minihack.envs:Retreat",
)
