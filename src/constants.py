from dataclasses import dataclass

LEARNING_RATE = .01
N_EPISODES = 100000
DISCOUNT_FACTOR = .9
MIN_EPSILON = .03
EPSILON_DELAY = .5


@dataclass(frozen=True)
class RewardSystem:
    on_success: float
    on_fail: float
    on_nothing: float = 0


BASE_REWARD_SYSTEM = RewardSystem(1, 0)
HOLE_PENALTY_REWARD_SYSTEM = RewardSystem(1, -1)
