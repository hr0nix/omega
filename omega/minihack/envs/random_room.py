from minihack.envs.room import MiniHackRoom
from gym.envs import registration


class MiniHackRoom7x7Random(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=7, random=True, **kwargs)


class MiniHackRoom9x9Random(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=9, random=True, **kwargs)


class MiniHackRoom11x11Random(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=11, random=True, **kwargs)


class MiniHackRoom13x13Random(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=13, random=True, **kwargs)


class MiniHackRoom14x14Random(MiniHackRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=14, random=True, **kwargs)


registration.register(
    id="MiniHack-Room-Random-7x7-v0",
    entry_point="omega.minihack.envs:MiniHackRoom7x7Random",
)


registration.register(
    id="MiniHack-Room-Random-9x9-v0",
    entry_point="omega.minihack.envs:MiniHackRoom9x9Random",
)


registration.register(
    id="MiniHack-Room-Random-11x11-v0",
    entry_point="omega.minihack.envs:MiniHackRoom11x11Random",
)


registration.register(
    id="MiniHack-Room-Random-13x13-v0",
    entry_point="omega.minihack.envs:MiniHackRoom13x13Random",
)

registration.register(
    id="MiniHack-Room-Random-14x14-v0",
    entry_point="omega.minihack.envs:MiniHackRoom14x14Random",
)
