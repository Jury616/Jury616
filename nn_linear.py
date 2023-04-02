import torch

torch.cuda.current_device()
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())


class Data(nn.Module):
    def __init__(self):
        super(Data, self).__init__()
        self.linear1 = torch.nn.Linear(196608, 10)
        # 这里的196608是图片展开为行向量后的维度

    def forward(self, input):
        output = self.linear1(input)
        return output


data = Data()
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)
for d in dataloader:
    imgs, targets = d
    print(imgs.shape)
    output = torch.flatten(imgs)
    # 将图片摊平为行向量
    output = data(output)
    print(output.shape)
