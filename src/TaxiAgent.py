from src.Agent import Agent, Observation
from gymnasium import Env
from dataclasses import dataclass


@dataclass(frozen=True)
class TaxiObservation:
    row: int
    col: int
    passenger: int
    destination: int

    @classmethod
    def from_int(cls, x: int):
        """Creates a new TaxiObservation by disassembling the int given by gymnasium"""
        dest = x % 4
        x //= 4
        passenger = x % 5
        x //= 5
        col = x % 5
        row = x // 5
        return cls(row, col, passenger, dest)


@dataclass(frozen=True)
class TaxiStepReport:
    transition_prob: int
    # todo
    # action_mask: ??


class TaxiAgent(Agent):
    def __init__(self, env: Env,
                 learning_rate: float,
                 n_episodes: int,
                 discount_factor: float,
                 epsilon_decay: float):
        super().__init__(env, learning_rate, n_episodes, discount_factor, epsilon_decay)

    def epoch(self, training: bool):
        ...

    def train(self):
        ...

    def calculate_reward(self, obs: Observation):
        ...
