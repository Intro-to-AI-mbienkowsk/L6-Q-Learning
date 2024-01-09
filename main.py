import argparse

import gymnasium as gym
from src.Agent import FrozenLakeAgent
from argparse import ArgumentParser
from src.constants import *


def run_model(args: argparse.Namespace):
    map_name = f"{args.s}x{args.s}"
    train_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=args.slip)
    test_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=args.slip, render_mode="human")
    agent = FrozenLakeAgent(env=train_env, reward_system=RewardSystem(*args.rs), learning_rate=args.lr,
                            n_episodes=args.e, discount_factor=args.d)
    agent.train()
    agent.env = test_env
    # todo: change to agent.test()
    agent.train()


def main():
    parser = ArgumentParser()
    parser.add_argument('-lr', type=float, default=LEARNING_RATE,
                        help='The learning rate of the model.')
    parser.add_argument('-e', type=int, default=N_EPISODES, help='Number of training episodes')
    parser.add_argument('-d', type=float, default=DISCOUNT_FACTOR,
                        help='The discount factor to use when training')
    parser.add_argument('-ed', type=float, help='Determines the speed of epsilon delay. After [e] * [ed] '
                                                f'episodes, the epsilon decays to the value of {MIN_EPSILON}')
    parser.add_argument('-s', type=int, choices=[4, 8], default=8, help='Grid size for the frozen lake problem.')
    parser.add_argument('-rs', type=tuple[float, 3], default=(1, 0, 0),
                        help='Reward system to use when training (a, b, c) where a  is rewarded on succesfully passsing'
                             ' the episode, b is given for falling into the hole and c in any other case.')
    parser.add_argument('-slip', type=bool, default=True, help='Determines whether the slippery (true) or'
                                                               'non-slippery map is used.')
    args = parser.parse_args()
    run_model(args)

    return 0


if __name__ == '__main__':
    main()
