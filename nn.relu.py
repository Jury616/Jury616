# relu非线性变换
import torch
import torchvision
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))


class data(torch.nn.Module):
    def __init__(self):
        super(data, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


data = data()
writer = SummaryWriter("logs_relu")
dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)
step = 0
for d in dataloader:
    imgs, targets = d
    writer.add_images("input", imgs, step)
    output = data(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
# 实现结果是图层output相当于出现了一层黑白滤镜，更好的实现了非线性方面的拟合
