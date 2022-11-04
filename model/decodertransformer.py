import torch.nn as nn
import torch
selfattention = nn.MultiheadAttention(embed_dim=32,num_heads=8,batch_first=True)
tensor = torch.rand((3,17,32))
query = torch.rand((3,16,32))
result = selfattention(key = tensor,value = tensor,query = query)
print(result[0].shape)
print(result[1].shape)
# decoder = nn.TransformerDecoderLayer(d_model=32,nhead=8,dim_feedforward=1024,batch_first=True)
# decodersequence = torch.rand((3,17,32))
# outputresult = decoder(decodersequence,decodersequence)
# print(outputresult.shape)

