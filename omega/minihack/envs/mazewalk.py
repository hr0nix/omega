from minihack.envs.mazewalk import MiniHackMazeWalk
from minihack.envs import register


class MiniHackMazeWalk11x11(MiniHackMazeWalk):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, w=11, h=11, premapped=False, **kwargs)


class MiniHackMazeWalk13x13(MiniHackMazeWalk):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, w=13, h=13, premapped=False, **kwargs)


register(
    id="MiniHack-MazeWalk-11x11-v0",
    entry_point="omega.minihack.envs.mazewalk:MiniHackMazeWalk11x11",
)

register(
    id="MiniHack-MazeWalk-13x13-v0",
    entry_point="omega.minihack.envs.mazewalk:MiniHackMazeWalk13x13",
)
