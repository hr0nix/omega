import os

from minihack.navigation import MiniHackNavigation
from minihack.envs import register


DES_PATH = os.path.join(os.path.dirname(__file__), 'des')


class ExploreRoomsBase(MiniHackNavigation):
    def __init__(self, des_file, max_episode_steps, *args, **kwargs):
        super().__init__(*args, des_file=des_file, max_episode_steps=max_episode_steps, **kwargs)


class ExploreRoomsEasy(ExploreRoomsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file=os.path.join(DES_PATH, 'explore_rooms_easy.des'),
            max_episode_steps=200,
            **kwargs
        )


register(
    id="MiniHack-ExploreRooms-Easy-v0",
    entry_point="omega.minihack.envs:ExploreRoomsEasy",
)
