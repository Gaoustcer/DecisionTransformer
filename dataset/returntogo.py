from torch.utils.data import Dataset

import d4rl
import gym
import numpy as np


class returntogodataset(Dataset):
    def __init__(self,load_from_file = False,loadpath = "rewardtogo.npy") -> None:
        super(returntogodataset,self).__init__()
        self.env = gym.make("maze2d-large-v1")
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.len = len(self.dataset['rewards'])
        self.transitionlen = 4
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

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    sequence()
    loader = DataLoader(returntogodataset(),batch_size=64)
    for states,actions,rewardtogo,timestep in loader:
        print(states.shape)
        print(actions.shape)
        print(rewardtogo.shape)
        print(timestep.shape)
        exit()
