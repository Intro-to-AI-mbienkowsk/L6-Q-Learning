from dataclasses import dataclass

LEARNING_RATE = .01
N_EPISODES = 100000
DISCOUNT_FACTOR = .9
MIN_EPSILON = .03


@dataclass(frozen=True)
class RewardSystem:
    on_fail: float
    on_success: float
    on_nothing: float = 0


BASE_REWARD_SYSTEM = RewardSystem(0, 1)
HOLE_PENALTY_REWARD_SYSTEM = RewardSystem(-1, 1)
