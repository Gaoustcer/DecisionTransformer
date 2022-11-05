import torch.nn as nn
import torch
attentionlayer = nn.MultiheadAttention(embed_dim=32,num_heads=8,batch_first=True)
tensor = torch.rand(3,17,32)
result, _ = attentionlayer(tensor,tensor,tensor)

print(result.shape)