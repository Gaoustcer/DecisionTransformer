import d4rl
import gym

env = gym.make('hopper-medium-v2')

import matplotlib.pyplot as plt

def returndistribution():
    dataset = d4rl.qlearning_dataset(env)
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    r_list = []
    Trajreturn = 0
    for r,done in zip(rewards,terminals):
        Trajreturn += r
        if done:
            r_list.append(Trajreturn)
            Trajreturn = 0

    plt.hist(r_list,bins=64)
    plt.savefig("distribution_hopper.png")

if __name__ == "__main__":
    returndistribution()