import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output


zrnn=NeuralNetwork()
x=torch.tensor(1.0)
print(zrnn(x))
