import torch.nn as nn
import gym


class Actioncopy(nn.Module):
    def __init__(self,env:gym.Env) -> None:
        super(Actioncopy,self).__init__()
        self.statedim = len(env.observation_space.sample())
        self.actiondim = len(env.action_space.sample())
        self.net = nn.Sequential(
            nn.Linear(self.statedim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,self.actiondim),
            nn.Tanh()
        )        


    def forward(self,states):
        return self.net(states)
        pass