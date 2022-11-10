import torch
import d4rl
import gym
from tqdm import tqdm

def dividedataset(envname = "hopper-medium-v2"):
    env = gym.make(envname)
    dataset = d4rl.qlearning_dataset(env)
    keys = ['observations','rewards','actions','terminals','next_observations']
    numberoftransitions = len(dataset['rewards'])
    def _check():
        for i in tqdm(range(numberoftransitions) - 1):
            if (dataset[keys[0]][i+1] == dataset[keys[-1]][i]).all() == False:
                pass
    trajinfo = []
    trajdict = dict()
    for key in keys:
        trajdict[key] = []
    for i in tqdm(range(numberoftransitions)):
        done = dataset['terminals'][i]
        for key in keys:
            trajdict[key].append(dataset[key][i])
        if done == True:
            trajinfo.append(trajdict)
            for key in keys:
                trajdict[key] = []
    import pickle
    with open("dataset.pkl",'wb') as fp:
        pickle.dump(trajinfo,fp)

import pickle
def loadfile():
    with open('dataset.pkl',"rb") as fp:
        dataset = pickle.load(fp)
    # print(dataset)
        return dataset 

def rewardtogo():
    # pass    
    dataset = loadfile()
    
    for item in tqdm(dataset):
        epsidelen = len(item['rewards'])
        item['rewardstogo'] = [0] * epsidelen
        # epsidelen = len(item['rewards'])
        result = 0
        for i in (range(epsidelen - 1,-1,-1)):
            result += item['rewards'][i]
            item['rewardstogo'][i] = result
    with open("datasetrewardstogo.pkl","wb") as fp:
        pickle.dump(dataset,fp)


if __name__ == "__main__":
    # dividedataset()
    # loadfile()
    rewardtogo()