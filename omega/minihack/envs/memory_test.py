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


class MiniHackMemoryTestEasy(MiniHackMemoryTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            des_file='memory_test_easy.des',
            **kwargs
        )


registration.register(
    id="MiniHack-CreditAssignmentTest-v0",
    entry_point="omega.minihack.envs:MiniHackCreditAssignmentTest",
)


registration.register(
    id="MiniHack-MemoryTestEasy-v0",
    entry_point="omega.minihack.envs:MiniHackMemoryTestEasy",
)
