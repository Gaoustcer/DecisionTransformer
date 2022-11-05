import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self) -> None:
        super(DecisionTransformer,self).__init__()
        self.embeddinglab