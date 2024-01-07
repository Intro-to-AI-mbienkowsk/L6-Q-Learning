from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from src.constants import *
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import Env
from typing import TypeVar

Observation = TypeVar('Observation')
Action = TypeVar('Action')


class Agent(ABC):
    def __init__(self, env: Env,
                 learning_rate: float = LEARNING_RATE,
                 n_episodes: int = N_EPISODES,
                 epsilon: float = EPSILON,
                 discount_factor: float = DISCOUNT_FACTOR):
        self.env = env
        self._lr = learning_rate
        self._n_episodes = n_episodes
        self._epsilon = epsilon
        # todo: qvalues setter
        self._q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self._discount_factor = discount_factor

    @property
    def learning_rate(self) -> float:
        return self._lr

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self._epsilon = value

    @property
    def q_values(self) -> defaultdict:
        return self._q_values

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_action(self, obs: Observation):
        if np.random.random() < self.epsilon:
            return self.get_random_action()
        return int(np.argmax(self.q_values[obs]))

    def update(self,
               obs: Observation,
               action: Action,
               reward: float,
               next_obs: Observation):
        best_next_q = np.max(self.q_values[next_obs])
        delta = reward + self.discount_factor * best_next_q - self.q_values[obs][action]
        self.q_values[obs][action] += self.learning_rate * delta


class FrozenLakeAgent(Agent):
    def __init__(self, env: Env):
        super().__init__(env)


@dataclass
class FrozenLakeStepReport:
    observation: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class FrozenLakeObservation:
    position: int


class Move(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
