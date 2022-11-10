from torch.utils.data import Dataset

import d4rl
import gym
import numpy as np


class returntogodataset(Dataset):
    def __init__(self,load_from_file = False,loadpath = "rewardtogo.npy",trajlen = 8) -> None:
        super(returntogodataset,self).__init__()
        self.env = gym.make("maze2d-large-v1")
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.len = len(self.dataset['rewards'])
        self.transitionlen = trajlen
        from tqdm import tqdm
        if load_from_file == False:
            # self.reward_to_go = [sum(self.dataset['rewards'][i:]) for i in tqdm(range(self.len))]
            result = 0
            self.reward_to_go = np.zeros(self.len)
            
            for i in tqdm(range(self.len - 1,-1,-1)):
                result += self.dataset['rewards'][i]
                self.reward_to_go[i] = result
            np.save("rewardtogo.npy",self.reward_to_go)
        else:
            self.reward_to_go = np.load(loadpath,allow_pickle=True)
        '''
        observations
        actions
        next_observations
        rewards
        terminals
        '''

    def __len__(self):
        return self.len - self.transitionlen
# np.arange()
    def __getitem__(self, index):
        return self.dataset['observations'][index:index + self.transitionlen],self.dataset['actions'][index:index + self.transitionlen],self.reward_to_go[index:index + self.transitionlen],np.arange(self.transitionlen)

        # return super().__getitem__(index) 

def sequence():
    env = gym.make("maze2d-large-v1")
    dataset = d4rl.qlearning_dataset(env)
    for i in range(len(dataset['rewards']) - 1):
        if (dataset['observations'][i+1] == dataset['next_observations'][i]).all() == False:
            print("alter")

from random import randint
import pickle
class Trajdataset(Dataset):
    def __init__(self,trajlengh = 16,path = "dataset/datasetrewardstogo.pkl") -> None:
        super().__init__()
        self.tralength = trajlengh
        with open(path,'rb') as fp:
            self.dataset = pickle.load(fp)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        index = randint(0,len(self.dataset[index]['rewardstogo']) - self.tralength)
        return self.dataset[index]['observations'][index:index+self.tralength],self.dataset[index]['actions'][index:index + self.tralength],self.dataset[index]['rewardstogo'][index:index + self.tralength],np.arange(0,self.tralength)
        # return super().__getitem__(index)

def averagereward():
    with open("datasetrewardstogo.pkl",'rb') as fp:
        datalist = pickle.load(fp)
    rewardlist = []
    for data in datalist:
        rewardlist.append(sum(data['rewards']))
    # print(rewardlist)
    print(sum(rewardlist)/len(rewardlist))

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch
    averagereward()
    exit()
    # sequence()
    loader = DataLoader(Trajdataset(path='datasetrewardstogo.pkl'),batch_size=4)
    for states,actions,rewardtogo,timestep in loader:
        # print("states is ",states)
        print("len of state",len(states))
        print("states 0 is",states[0].shape)
        states = torch.stack(states,dim=1)
        print(states.shape)
        actions = torch.stack(actions,dim=1)
        print(actions.shape)
        rewardtogo = torch.stack(rewardtogo,dim=1).unsqueeze(-1)
        print("reward to go is",rewardtogo)
        print(rewardtogo.shape)
        print(timestep.shape)
        exit()
