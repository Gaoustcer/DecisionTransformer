import d4rl
import torch
import gym
from dataset.returntogo import returntogodataset
from model.positionembedding import PositionEmbedding
from model.actionprediction import DecisionTransformer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model.actioncopy import Actioncopy
from dataset.returntogo import Trajdataset
# import wandb
# wandb.init(project="DecisionTransformer")
# wandb.init(
#     confi
# )
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
        self.validateepoch = 32


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


# class 
class DecisionSeq(object):
    def __init__(self,envname = "hopper-medium-v2",trajlen = 16,datasetTraj = None) -> None:
        self.envname = envname
        self.env = gym.make(envname)
        self.Transformer = DecisionTransformer(self.env).cuda()
        self.Actioncopy = Actioncopy(self.env).cuda()
        self.trajlen = trajlen
        self.writer = SummaryWriter("./logs/TransformerDecision{}".format(envname))
        self.actioncopyoptim = torch.optim.Adam(self.Actioncopy.parameters(),lr = 0.0001)
        if datasetTraj is None:
            self.loader = DataLoader(Trajdataset(trajlengh=self.trajlen),batch_size = 32)
        else:
            self.loader = DataLoader(returntogodataset(load_from_file=True,trajlen=self.trajlen),batch_size=32)
        self.optim = torch.optim.Adam(self.Transformer.parameters(),lr = 0.0001)
        self.validatetime = 32
        self.actiondim = len(self.env.action_space.sample())
        self.statedim = len(self.env.observation_space.sample())
        self.index = 0

    def train(self):
        for states,actions,rewardstogo,timestep in tqdm(self.loader):
            if self.index % 32 == 0:
                r = self.validate()
                self.writer.add_scalar("reward",r,self.index//32)
            if isinstance(states,list):
                states = torch.stack(states,dim=1).cuda().to(torch.float32)
                actions = torch.stack(actions,dim=1).cuda().to(torch.float32)
                rewardstogo = torch.stack(rewardstogo,dim=1).cuda().unsqueeze(-1).to(torch.float32)
            else:
                states = states.cuda().to(torch.float32)
                actions = actions.cuda().to(torch.float32)
                rewardstogo = rewardstogo.cuda().to(torch.float32).unsqueeze(-1)
            timestep = timestep.cuda()
            predactions,embedding = self.Transformer(states,actions,rewardstogo,timestep)
            self.optim.zero_grad()
            loss = torch.nn.functional.mse_loss(predactions,actions)
            
            loss.backward()
            self.writer.add_scalar("loss",loss,self.index)
            self.index += 1
            # self.index % 32 == 0:

            self.optim.step()
        # pass       

    def actioncopytrain(self):
        for states,actions,_,_ in tqdm(self.loader):
            # print("states is",states)
            states = torch.stack(states,dim=1)
            actions = torch.stack(actions,dim=1).cuda()
            # print("state shape",states.shape)
            states = states.cuda()
            predactions = self.Actioncopy(states)
            self.actioncopyoptim.zero_grad()
            loss = torch.nn.functional.mse_loss(predactions,actions)
            loss.backward()
            self.writer.add_scalar("loss",loss,self.index)
            self.index += 1

            self.actioncopyoptim.step()
        pass

    def actioncopyvalidate(self):
        reward = 0
        for _ in range(self.validatetime):
            done = False
            state = self.env.reset()
            while done == False:
                action = self.Actioncopy(torch.from_numpy(state).cuda().to(torch.float32)).detach().cpu().numpy()
                ns,r,done,_ = self.env.step(action)
                state = ns
                reward += r
        return reward/self.validatetime


    def save(self,r):
        torch.save(self.Transformer,f"model/Transformer_withrewrd{r}_env_{self.envname}")
    def validate(self):
        reward = 0
        from tqdm import tqdm
        for _ in (range(self.validatetime)):
            # exptectreward = 300
            # rewardtogo = 79445
            rewardtogo = 187.7212032675743
            stateslist = []
            actionslist = []
            rewardstogolist = []
            done = False
            state = self.env.reset()
            actionslist.append(torch.rand(self.actiondim).cuda())
            rewardstogolist.append(rewardtogo)
            stateslist.append(torch.from_numpy(state).cuda())
            timestep = [0]
            while done == False:
                if len(timestep) <= self.trajlen:
                    states = torch.stack(stateslist).unsqueeze(0).to(torch.float32)
                    actions = torch.stack(actionslist).unsqueeze(0).to(torch.float32)
                    rewards = torch.tensor(rewardstogolist).cuda().unsqueeze(0).unsqueeze(-1).to(torch.float32)
                    predactions,embedding = self.Transformer.forward(states,actions,rewards,torch.tensor(timestep).unsqueeze(0).cuda())
                    timestep.append(timestep[-1] + 1)
                    # print("pred actions",predactions,predactions.shape)
                    action = predactions[:,-1].squeeze().detach().cpu().numpy()
                    ns,r,done,_ = self.env.step(action)
                    stateslist.append(torch.from_numpy(ns).cuda())
                    actionslist[-1] = torch.from_numpy(action).cuda()
                    actionslist.append(torch.rand(self.actiondim).cuda())
                    rewardstogolist.append(rewardstogolist[-1] - r)
                    reward += r

                else:
                    stateslist = stateslist[-self.trajlen:]
                    actionslist = actionslist[-self.trajlen:]
                    rewardstogolist = rewardstogolist[-self.trajlen:]
                    timestep = timestep[:self.trajlen]
                    
        return reward/self.validatetime


# def 


if __name__ == "__main__":
    # DecisonModel = DecisionSeq(envname = "maze2d-large-v1",datasetTraj=1)
    DecisonModel = DecisionSeq()
    # print(DecisonModel.actioncopyvalidate())
    # exit()
    # reward = DecisonModel.validate()
    # print("reward is ",reward)
    # exit()
    # main = Agent()
    # main.train()
    # loader = DataLoader(returntogodataset(load_from_file=True),batch_size=32)
    # for element in loader:
    #     for item in element:
    #         print("item is",item.shape)
    #     exit()

    EPOCH = 16
    for epoch in range(EPOCH):
        DecisonModel.actioncopytrain()
        r = DecisonModel.actioncopyvalidate()
        print("epoch {}".format(epoch),"reward {}".format(r))
    exit()
    for epoch in range(EPOCH):
        DecisonModel.train()
        reward = DecisonModel.validate()
        print("reward for epoch{}".format(epoch),reward)

    r = DecisonModel.validate()
    DecisonModel.save(int(r))
            
        


        