from minihack.envs.mazewalk import MiniHackMazeWalk
from minihack.envs import register


class MiniHackMazeWalk11x11(MiniHackMazeWalk):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, w=11, h=11, premapped=False, **kwargs)


class MiniHackMazeWalk12x12(MiniHackMazeWalk):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, w=12, h=12, premapped=False, **kwargs)


class MiniHackMazeWalk13x13(MiniHackMazeWalk):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, w=13, h=13, premapped=False, **kwargs)


class MiniHackMazeWalk14x14(MiniHackMazeWalk):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        super().__init__(*args, w=14, h=14, premapped=False, **kwargs)


register(
    id="MiniHack-MazeWalk-11x11-v0",
    entry_point="omega.minihack.envs.mazewalk:MiniHackMazeWalk11x11",
)


register(
    id="MiniHack-MazeWalk-12x12-v0",
    entry_point="omega.minihack.envs.mazewalk:MiniHackMazeWalk11x11",
)


register(
    id="MiniHack-MazeWalk-13x13-v0",
    entry_point="omega.minihack.envs.mazewalk:MiniHackMazeWalk13x13",
)


register(
    id="MiniHack-MazeWalk-14x14-v0",
    entry_point="omega.minihack.envs.mazewalk:MiniHackMazeWalk14x14",
)
