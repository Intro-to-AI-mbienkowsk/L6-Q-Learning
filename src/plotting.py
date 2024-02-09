import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number
import numpy as np

from src.constants import RewardSystem


@dataclass(frozen=True)
class AgentInfo:
    reward_system: RewardSystem
    n_episodes: int
    discount_factor: float

    def __str__(self):
        description = "Agent's parameters:\n"
        description += (f'Reward system (W/L): ({self.reward_system.on_success}/'
                        f'{0 if self.reward_system.on_fail == 0.0 else self.reward_system.on_fail})\n')
        description += f'Number of episodes: {self.n_episodes}\n'
        description += f'Discount factor: {self.discount_factor}'
        return description


def plot_agent_data(info: AgentInfo, plot_data: list[Number], title: str):
    fig = plt.figure(figsize=(12, 6))
    fig.set_facecolor('#f8e5ad')
    plt.subplots_adjust(right=.67, left=.075)
    plt.plot([i for i in range(len(plot_data))], plot_data)
    plt.text(1.1, 0.5, str(info), transform=plt.gca().transAxes, va='center', fontsize=12)
    plt.title(title)
    plt.show()


def visualize_q_values(avg_q_vals: dict):
    mean_array = np.array([[avg_q_vals[8 * i + j] for j in range(8)] for i in range(8)])
    plt.imshow(mean_array, cmap='viridis', vmin=np.min(mean_array), vmax=np.max(mean_array))
    plt.title("Visualization of the agent's q-values")
    plt.colorbar()
    plt.show()
