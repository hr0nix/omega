import os.path

from gym.envs import registration
from minihack import MiniHackNavigation, RewardManager


DES_PATH = os.path.join(os.path.dirname(__file__), 'des')


class MiniHackMemoryTestBase(MiniHackNavigation):
    def __init__(self, des_file, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        reward_manager = RewardManager()
        reward_manager.add_message_event(
            ["squeak"],
            reward=-1,
            terminal_sufficient=True,
            terminal_required=True,
        )
        reward_manager.add_message_event(
            ["fall"],
            reward=1,
            terminal_sufficient=True,
            terminal_required=True,
        )
        super().__init__(
            *args,
            des_file=os.path.join(DES_PATH, des_file),
            reward_manager=reward_manager,
            **kwargs
        )


class MiniHackCreditAssignmentTest(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file='credit_assignment_test.des',
            **kwargs
        )


class MiniHackMemoryTest4Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_4_steps.des', **kwargs)


class MiniHackMemoryTest5Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_5_steps.des', **kwargs)


class MiniHackMemoryTest6Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_6_steps.des', **kwargs)


class MiniHackMemoryTest7Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_7_steps.des', **kwargs)


class MiniHackMemoryTest8Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_8_steps.des', **kwargs)


class MiniHackMemoryTest9Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_9_steps.des', **kwargs)


class MiniHackMemoryTest10Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_10_steps.des', **kwargs)


class MiniHackMemoryTest11Steps(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, des_file='memory_test_11_steps.des', **kwargs)


registration.register(
    id="MiniHack-CreditAssignmentTest-v0",
    entry_point="omega.minihack.envs:MiniHackCreditAssignmentTest",
)


registration.register(
    id="MiniHack-MemoryTest-4-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest4Steps",
)


registration.register(
    id="MiniHack-MemoryTest-5-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest5Steps",
)


registration.register(
    id="MiniHack-MemoryTest-6-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest6Steps",
)


registration.register(
    id="MiniHack-MemoryTest-7-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest7Steps",
)


registration.register(
    id="MiniHack-MemoryTest-8-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest8Steps",
)


registration.register(
    id="MiniHack-MemoryTest-9-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest9Steps",
)


registration.register(
    id="MiniHack-MemoryTest-10-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest10Steps",
)


registration.register(
    id="MiniHack-MemoryTest-11-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTest11Steps",
)
