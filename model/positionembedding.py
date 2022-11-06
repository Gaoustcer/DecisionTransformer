import torch.nn as nn
import numpy as np
import torch



class PositionEmbedding(nn.Module):
    def __init__(self,embedding_dim=32,max_len = 64) -> None:
        super(PositionEmbedding,self).__init__()
        self.timestepembedding = nn.Embedding(num_embeddings = max_len,embedding_dim = embedding_dim)
        # self.batch_size = 64
        # self.mask = torch.tril(torch.ones(self.batch_size,self.batch_size),diagonal=0)
        import d4rl
        import gym
        self.testenv = gym.make('maze2d-large-v1')
        self.statedim = len(self.testenv.observation_space.sample())
        self.actiondim = len(self.testenv.action_space.sample())
        self.embeddim = embedding_dim
        self.stateembedding = nn.Sequential(
            nn.Linear(self.statedim,32),
            nn.ReLU(),
            nn.Linear(32,embedding_dim)
        )
        self.actionembedding = nn.Sequential(
            nn.Linear(self.actiondim,32),
            nn.ReLU(),
            nn.Linear(32,embedding_dim)
        ) 
        self.rewardembedding = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,embedding_dim)
        )
        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = 8,
           dim_feedforward = 128,
           batch_first = True
        )
        self.actionprediction = nn.Sequential(
            nn.Linear(embedding_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,self.actiondim),
            nn.Tanh()
        )
    
    def get_action(self,states,actions,rewardstogo,timesteps):
        # states [N,4]
        # actions [N - 1 ,2]
        # rewardtogo [N,1]
        # timestep [N]
        timestepsembedding = self.timestepembedding(timesteps)
        # print("states is",states.shape)
        stateembedding = self.stateembedding(states) + timestepsembedding
        rewardembedding = self.rewardembedding(rewardstogo) + timestepsembedding
        sequencelen = states.shape[0]
        mask = torch.tril(torch.ones(3 * sequencelen - 1,3 * sequencelen - 1),diagonal=0).cuda()
        if actions is not None:
            # print("embedding action dim is",actions.shape)
            # print("timestep is",timestepsembedding.shape)
            # if timestepsembedding.shape[0] != actions.shape[0]:
            actionembedding = self.actionembedding(actions) + timestepsembedding[:-1]
            # else:

            # print(actionembedding)
            # print(stateembedding[:-1].shape,rewardembedding[:-1].shape,actionembedding[:-1].shape)
            embedding = torch.concat((stateembedding[:-1],rewardembedding[:-1],actionembedding),dim=-1)
            embedding = embedding.reshape(3 * (sequencelen - 1),self.embeddim)
            newstateacion = torch.stack((stateembedding[-1],rewardembedding[-1]),dim=0)
            embedding = torch.concat((embedding,newstateacion),dim=0)
        else:
            embedding = torch.concat([stateembedding,rewardembedding],-1)
            embedding = embedding.reshape(2,32)
            '''
            only a state and action reshape N = 1
            '''
        predactions = self.transformer(src = embedding,tgt = embedding,src_mask = mask,tgt_mask = mask)
        return self.actionprediction(predactions[-1])


    def forward(self,states,actions,rewardstogo,timesteps):
        timestepembedding = self.timestepembedding(timesteps)
        stateembedding = self.stateembedding(states) + timestepembedding
        actionembedding = self.actionembedding(actions) + timestepembedding
        # if  is not None:
        rewardembedding = self.rewardembedding(rewardstogo) + timestepembedding
        embedding = torch.concat(
            [stateembedding,actionembedding,rewardembedding],
            dim = -1
        )
        # else:
        #     embedding = torch.concat([
        #         stateembedding,actionembedding],
        #         dim=-1
        #     )
        # embedding = embedding.reshape()
        # if len(states.shape) == 2:
        #     # states (L,E)
        #     # concat embedding (L,3E),reshape as (3L,E)
        #     self.sequencelen = states.shape[0]
        #     embedding = embedding.reshape(3 * self.sequencelen,self.embeddim)
        # elif len(states.shape) == 3:
            # states (N,L,E)
        self.sequencelen = states.shape[1]
        embedding = embedding.reshape(states.shape[0], 3 * self.sequencelen,self.embeddim)
        # print("batchsize is",self.batchsize)
        self.sequencelen *= 3
        # print("embedding shape is",embedding.shape)
        self.mask = torch.tril(torch.ones(self.sequencelen,self.sequencelen),diagonal=0).cuda()
        actionembedding = self.transformer(embedding,embedding,src_mask = self.mask,tgt_mask = self.mask)
        # print(actionembedding.shape)
        # actionembedding = actionembedding[]
        if len(states.shape) == 2:
            actionembedding = actionembedding[1::3,:]
        elif len(states.shape) == 3:
            actionembedding = actionembedding[:,1::3,:]
        return self.actionprediction(actionembedding)
            # states (L,E),L is the length of states

# if __name__ == "__main__":
    