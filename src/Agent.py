from collections import defaultdict
from enum import Enum
from itertools import accumulate
from src.constants import *
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import Env
from typing import TypeVar
import random

from src.plotting import plot_agent_data, AgentInfo

Observation = TypeVar('Observation')
Action = TypeVar('Action')


@dataclass(frozen=True)
class FrozenLakeStepReport:
    position: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass(frozen=True)
class FrozenLakeObservation:
    position: int
    info: dict


class Agent(ABC):
    def __init__(self, env: Env,
                 learning_rate: float,
                 n_episodes: int,
                 discount_factor: float,
                 epsilon_decay: float):
        self._env = env
        self._lr = learning_rate
        self._n_episodes = n_episodes
        self._epsilon = 1
        self._epsilon_decay_rate = 1 / (self.n_episodes * epsilon_decay)
        self._q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self._discount_factor = discount_factor

    @property
    def env(self) -> Env:
        return self._env

    @env.setter
    def env(self, value: Env):
        self._env = value

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

    @q_values.setter
    def q_values(self, val) -> None:
        self._q_values = val

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def epsilon_decay_rate(self) -> float:
        return self._epsilon_decay_rate

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_action(self, obs: Observation):
        if np.random.random() < self.epsilon:
            return self.get_random_action()
        return random.choice([action for action in range(0, self.env.action_space.n)
                              if self.q_values[obs][action] == np.max(self.q_values[obs])])

    def update(self,
               pos: int,
               action: Action,
               reward: float,
               next_pos: int):
        best_next_q = np.max(self.q_values[next_pos])
        delta = reward + self.discount_factor * best_next_q - self.q_values[pos][action]
        self.q_values[pos][action] += self.learning_rate * delta

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, MIN_EPSILON)

    @abstractmethod
    def epoch(self, training: bool):
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def calculate_reward(self, obs: Observation):
        ...


class FrozenLakeAgent(Agent):
    def __init__(self, env: Env,
                 reward_system: RewardSystem = BASE_REWARD_SYSTEM,
                 learning_rate: float = LEARNING_RATE,
                 n_episodes: int = N_EPISODES,
                 discount_factor: float = DISCOUNT_FACTOR,
                 epsilon_decay: float = .5
                 ):
        super().__init__(env, learning_rate=learning_rate, n_episodes=n_episodes, discount_factor=discount_factor,
                         epsilon_decay=epsilon_decay)
        self.reward_system = reward_system
        self.epoch_log = []

    def epoch(self, training: bool):
        obs = FrozenLakeObservation(*self.env.reset())
        terminal_state = False
        while not terminal_state:
            action = self.get_action(obs.position)
            report = FrozenLakeStepReport(*self.env.step(action))
            reward = self.calculate_reward(report)
            if training:
                self.update(obs.position, action, reward, report.position)
            terminal_state = report.terminated or report.truncated
            obs = report
        self.epoch_log.append(reward := self.calculate_reward(obs))
        # return the report with the custom reward injected
        return FrozenLakeStepReport(obs.position, reward, obs.terminated, obs.truncated, obs.info)

    def calculate_reward(self, obs: FrozenLakeStepReport) -> float:
        """ Calculates the reward for a given move that conforms to self.reward_system
         based on the outcome of the move"""
        if obs.truncated:
            return self.reward_system.on_fail
        if obs.terminated:
            return self.reward_system.on_success if obs.reward > 0 else self.reward_system.on_fail
        return self.reward_system.on_nothing

    def train(self):
        passed = 0
        for e in range(self.n_episodes):
            obs = self.epoch(training=True)
            self.decay_epsilon()
            passed += obs.reward

    def test(self, n_test_epochs):
        """Returns the fraction succesfull_epochs/n_test_epochs for a given test env"""
        succesfull_epochs = 0
        for i in range(n_test_epochs):
            succesfull_epochs += self.epoch(training=False).reward == self.reward_system.on_success
        return round(succesfull_epochs / n_test_epochs, 2)

    def plot_goal_function_in_time(self):
        """Plots how the sum of the goal function changed over the training epochs"""
        info = AgentInfo(self.reward_system, self.n_episodes, self.discount_factor)
        plot_agent_data(info, self.epoch_log, "Reward function across the training epochs")

    def plot_goal_function_sum(self):
        """Plots how the sum of the goal function changed over the training epochs"""
        info = AgentInfo(self.reward_system, self.n_episodes, self.discount_factor)
        plot_agent_data(info, list(accumulate(self.epoch_log)), "Sum of the reward function across the training epochs")

    def calculate_average_q_values(self):
        """Returns dict idx: q_val where q_val is the average q-value for a move from a field with the idx"""
        return {field_idx: np.average(self.q_values[field_idx]) for field_idx in range(64)}


class Move(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
