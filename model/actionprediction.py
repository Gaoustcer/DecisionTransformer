import torch
import torch.nn as nn
import gym
class DecisionTransformer(nn.Module):
    def __init__(self,env:gym.Env,embed_dim = 32) -> None:
        super(DecisionTransformer,self).__init__()
        # self.embeddinglab
        self.statedim = len(env.observation_space.sample())
        self.embeddim = embed_dim
        self.actiondim = len(env.action_space.sample())
        self.rewardnet = nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,embed_dim)
        )
        self.statenet = nn.Sequential(
            nn.Linear(self.statedim,32),
            nn.ReLU(),
            nn.Linear(32,self.embeddim)
        )
        self.actionnet = nn.Sequential(
            nn.Linear(self.actiondim,32),
            nn.ReLU(),
            nn.Linear(32,self.embeddim)
        )
        self.attentionlayer = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=8,batch_first=True)
        self.actionpred = nn.Sequential(
            nn.Linear(embed_dim,32),
            nn.ReLU(),
            nn.Linear(32,self.actiondim)
        )
        self.timestepembedding = nn.Embedding(num_embeddings=1024,embedding_dim=embed_dim)

    def forward(self,states,actions,rewards,timesteps):
        '''
        states [N,L,sdim]
        actions [N,L,adim]
        rewards [N,L,1]
        timestep [N,L,1]
        '''
        timestepembed = self.timestepembedding(timesteps)
        statesembed = self.statenet(states) + timestepembed
        actionsembed = self.actionnet(actions) + timestepembed
        rewardsembed = self.rewardnet(rewards) + timestepembed
        batchnum = states.shape[0]
        squencelen = states.shape[1]

        embedding = torch.reshape(
            torch.concat((rewardsembed,statesembed,actionsembed),dim=-1),
            (batchnum,3 * squencelen,self.embeddim)
        )
        mask = torch.tril(torch.ones(3 * squencelen, 3 * squencelen),diagonal = 0).cuda()
        transformer = self.attentionlayer(key = embedding,value = embedding,query = embedding,need_weights = False,attn_mask = mask)[0]
        # print("transformer is",transformer)
        return self.actionpred(transformer[:,1::3]),embedding

if __name__ == "__main__":
    import d4rl

    # import gym
    env = gym.make("hopper-medium-v2")
    states = torch.randn(3,17,11).cuda()
    actions = torch.randn(3,17,3).cuda()
    rewards = torch.randn(3,17,1).cuda()
    net = DecisionTransformer(env).cuda()
    predactions = net(states,actions,rewards)
    print(predactions.shape)

