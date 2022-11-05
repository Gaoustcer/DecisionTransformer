import torch.nn as nn
import numpy as np




class PositionEmbedding(nn.Module):
    def __init__(self,embedding_dim=32,max_len = 64) -> None:
        super(PositionEmbedding,self).__init__()
        self.timestepembedding = nn.Embedding(num_embeddings = max_len,embedding_dim = embedding_dim)
                