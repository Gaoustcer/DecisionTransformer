from dataset.returntogo import returntogodataset
from torch.utils.data import DataLoader
from model.positionembedding import PositionEmbedding
import torch
if __name__ == "__main__":
    loader = DataLoader(returntogodataset(),batch_size=64)
    net = PositionEmbedding().cuda()
    states = torch.rand((2,4)).cuda()
    actions = torch.rand((1,2)).cuda()
    reward = torch.rand(2,1).cuda()
    timestep = torch.tensor([0,1]).cuda()
    predaction = net.get_action(states,actions,reward,timestep)
    print(predaction.shape)
    # for states,actions,rewardstogo,timesteps in loader:
    #     states = states.cuda()
    #     actions = actions.cuda()
    #     rewardstogo = rewardstogo.cuda().unsqueeze(-1).to(torch.float32)
    #     timesteps = timesteps.cuda()
    #     newactions = net(states,actions,rewardstogo,timesteps)
    #     print("newactions",newactions.shape)
    #     exit()