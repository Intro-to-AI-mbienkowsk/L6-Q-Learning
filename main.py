import argparse

import gymnasium as gym
from src.Agent import FrozenLakeAgent
from argparse import ArgumentParser
from src.constants import *
from src.plotting import visualize_q_values


def run_model(args: argparse.Namespace):
    map_name = f"{args.s}x{args.s}"
    train_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=args.slip)
    visible_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=args.slip, render_mode="human")
    agent = FrozenLakeAgent(env=train_env, reward_system=RewardSystem(args.on_win, args.on_lose, args.on_else),
                            learning_rate=args.lr, n_episodes=args.e, discount_factor=args.d, epsilon_decay=args.ed)
    agent.train()
    print(f"During the test, the agent had a success rate of {agent.test(1000)}")

    agent.env = visible_env
    if args.display:
        print("The agent will now attempt to solve the problem 3 times with the environment visible"
              " to you. To quit, press Ctrl+C")
        agent.test(3)
    if args.visualize:
        visualize_q_values(agent.calculate_average_q_values())
    if args.goal_sum:
        agent.plot_goal_function_sum()


def main():
    parser = ArgumentParser()
    parser.add_argument('-lr', type=float, default=LEARNING_RATE,
                        help='The learning rate of the model.')
    parser.add_argument('-e', type=int, default=N_EPISODES, help='Number of training episodes')
    parser.add_argument('-d', type=float, default=DISCOUNT_FACTOR,
                        help='The discount factor to use when training')
    parser.add_argument('-ed', type=float, help='Determines the speed of epsilon delay. After [e] * [ed] '
                                                f'episodes, the epsilon decays to the value of {MIN_EPSILON}', default=.5)
    parser.add_argument('-s', type=int, choices=[4, 8], default=8, help='Grid size for the frozen lake problem.')
    parser.add_argument('-on_win', type=float, default=1,
                        help='Reward given to the agent after a succesful episode')
    parser.add_argument('-on_lose', type=float, default=-1,
                        help='Reward given to the agent after an unsuccesful episode')
    parser.add_argument('-on_else', type=float, default=0,
                        help='Reward given to the agent after any non-episode terminating move')
    parser.add_argument('-slip', type=int, choices=[0, 1], default=1, help='Determines whether the slippery (1) or'
                                                                           'non-slippery map is used.')
    parser.add_argument('-visualize', nargs='?', const=True, default=True,
                        help='Visualize the q-table with a heatmap after training')
    parser.add_argument('-display', nargs='?', const=True, default=False,
                        help='Displays the environment and shows the agent complete 3 episodes.')
    parser.add_argument('-goal_sum', nargs='?', const=True, default=False,
                        help='Show the plot portraying how the sum of the ' +
                             'agent\'s goal function has changed over the time.')
    args = parser.parse_args()
    run_model(args)
    return 0


if __name__ == '__main__':
    main()
