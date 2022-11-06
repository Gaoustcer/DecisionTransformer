import torch.nn as nn
import torch
# attentionlayer = nn.MultiheadAttention(embed_dim=32,num_heads=8,batch_first=True)
# tensor = torch.rand(3,17,32)
# result, _ = attentionlayer(tensor,tensor,tensor)
transformer = nn.Transformer(
    d_model=32,nhead=8,
    batch_first=True
)
states = torch.rand((1,2,32))
actions = torch.rand((1,2,32))
rewards = torch.rand((1,2,32))
embedding = torch.stack((states,actions,rewards),dim=1).permute(0,2,1,3).reshape(1,3 * 2,32)
print("embedding shape is",embedding.shape)
result = transformer(embedding,embedding)
print(result.shape)
# print(result.shape)