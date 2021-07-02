import torch
from torch import nn


class Cc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output


cc=Cc()
x=torch.tensor(1.0)
output=cc(x)
print(output)
