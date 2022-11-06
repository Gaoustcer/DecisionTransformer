import d4rl
import torch
import gym
from dataset.returntogo import returntogodataset
from model.positionembedding import PositionEmbedding
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

class Agent(object):
    def __init__(self) -> None:
        self.transformer = PositionEmbedding().cuda()
        self.dataset = returntogodataset(load_from_file=True,loadpath="dataset/rewardtogo.npy",trajlen = 16)
        self.EPOCH = 32
        self.loader = DataLoader(self.dataset,batch_size = 64)
        self.optim = torch.optim.Adam(self.transformer.parameters(), lr = 0.0003)
        self.writer = SummaryWriter("./logs/DT")
        self.lossindex = 0
        self.trajlen = 16
        self.validateindex = 0
        self.env = gym.make('maze2d-large-v1')
        self.validateepoch = 2


    def train(self):
        for epoch in range(self.EPOCH):
            for states,actions,rewardstogo,timesteps in tqdm(self.loader):
                states = states.cuda()
                actions = actions.cuda()
                rewardstogo = rewardstogo.cuda().unsqueeze(-1).to(torch.float32)
                timesteps = timesteps.cuda()
                constuctionactions = self.transformer.forward(states,actions,rewardstogo,timesteps)
                actionloss = F.mse_loss(constuctionactions,actions)
                self.optim.zero_grad()
                actionloss.backward()
                self.optim.step()
                self.writer.add_scalar("loss",actionloss,self.lossindex)
                self.lossindex += 1
                if self.lossindex % 1024 == 0:
                    self.validate()
    
    def validate(self):
        Expertreward = 300
        realreward = 0

        for _ in range(self.validateepoch):
            done = False
            statelist = []
            actionlist = []
            rewardtogolist = [Expertreward]
            state = self.env.reset()
            timestep = [0]
            while done == False:
                statelist.append(torch.from_numpy(state).to(torch.float32).cuda())
                if len(timestep) < self.trajlen:
                    rewardtogo = torch.tensor(rewardtogolist).cuda().unsqueeze(-1).to(torch.float32)
                    states = torch.stack(statelist)
                    actions = None if len(actionlist) == 0 else torch.stack(actionlist)
                    # print("actions is ",actions)
                    nextaction = self.transformer.get_action(
                        states = states,
                        actions = actions,
                        rewardstogo = rewardtogo,
                        timesteps = torch.tensor(timestep).cuda()
                    )
                    # mask = torch.tril(torch.ones(len(timestep),len(timestep)),diagonal=0).cuda()
                    # action = self.transformer.forward(states=states,
                    # rewardstogo=rewardtogo,
                    # timesteps=torch.tensor(timestep).cuda(),
                    # actions=None if len(actionlist) == 0 else torch.stack(actionlist).cuda().to(torch.float32))
                else:
                    rewardtogo = torch.tensor(rewardtogolist[-self.trajlen:]).cuda().unsqueeze(-1).to(torch.float32)
                    states = torch.stack(statelist[-self.trajlen:])
                    actions = torch.stack(actionlist[-(self.trajlen)+1:])
                    # print("actions shape is",actions.shape)
                    nextaction = self.transformer.get_action(
                        states = states,
                        actions = actions,
                        rewardstogo = rewardtogo,
                        timesteps = torch.tensor(timestep).cuda()
                    )
                    # action = self.transformer.forward(states=states,
                    # rewardstogo=rewardtogo,
                    # actions=torch.stack(actionlist).cuda().to(torch.float32),
                    # timesteps=torch.from_numpy(np.arange(0,self.trajlen)))
                actionlist.append(nextaction)
                ns,r,done,_ = self.env.step(nextaction.cpu().detach().numpy())
                state = ns
                rewardtogolist.append(rewardtogolist[-1] - r)
                realreward += r
                if len(timestep) < self.trajlen:
                    timestep.append(timestep[-1] + 1)
        self.writer.add_scalar("return",realreward/self.validateepoch,self.validateindex)
        self.validateindex += 1
        return realreward/self.validateepoch

                        

if __name__ == "__main__":
    main = Agent()
    main.train()

                
            
        


        