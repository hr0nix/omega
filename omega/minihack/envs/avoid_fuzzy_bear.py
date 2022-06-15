import os.path

from gym.envs import registration
from minihack import MiniHackNavigation, RewardManager


DES_PATH = os.path.join(os.path.dirname(__file__), 'des')


class AvoidFuzzyBear(MiniHackNavigation):
    """
    Follows the idea of "Avoid fuzzy bear" environment from "Causally Correct Partial Models"
    https://arxiv.org/pdf/2002.02836v1.pdf
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        reward_manager = RewardManager()
        reward_manager.add_message_event(
            msgs=["squeak"],
            reward=1.0,
            terminal_sufficient=True,
            terminal_required=True,
        )
        reward_manager.add_message_event(
            msgs=["fall"],
            reward=-0.5,
            terminal_sufficient=True,
            terminal_required=True,
        )
        reward_manager.add_coordinate_event(
            # This is actually (5, 1) in the coordinate system of the des file.
            coordinates=(5, 4),
            reward=0.6,
            terminal_sufficient=True,
            terminal_required=True,
        )
        super().__init__(
            *args,
            des_file=os.path.join(DES_PATH, 'avoid_fuzzy_bear.des'),
            reward_manager=reward_manager,
            **kwargs
        )


registration.register(
    id="MiniHack-AvoidFuzzyBear-v0",
    entry_point="omega.minihack.envs:AvoidFuzzyBear",
)
