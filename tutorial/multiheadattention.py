import torch
import torch.nn as nn

multiheadattention = nn.MultiheadAttention(embed_dim=32,num_heads=8,batch_first=True)
vector = torch.randn(1,5,32)
first = vector[:,0,:].unsqueeze(0)
print("first is",first)

result = multiheadattention(first,first,first)
print(result[0][:,0])
mask = torch.triu(torch.ones(5,5),diagonal=0)
# t
print(mask)
result = multiheadattention(vector,vector,vector,attn_mask = mask)

print(result[0][:,0])
