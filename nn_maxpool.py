import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input_= torch.tensor([[1,2,0,3,1],
                       [0,1,2,3,1],
                       [1,2,1,0,0],
                       [5,2,3,1,1],
                       [2,1,0,1,1]],dtype=torch.float32)
input_=torch.reshape(input_,(-1,1,5,5))
print(input_.shape)


class data(nn.Module):
    def __init__(self):
        super(data,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)
        # dilation是取kernel和input重合部分隔x单位取，ceil_mode=True时会保留kernel出界时的部分
        # 最大池化获得的是重合部分的最大值，可以减少参数，加快训练的参数，同时也可以对数据进行裁剪

    def forward(self,input):
        output=self.maxpool1(input)
        return output


data=data()
writer=SummaryWriter("logs_maxpool")
dataset=torchvision.datasets.CIFAR10("./dataset",train=False,
                                       transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64)
step=0
for d in dataloader:
    imgs,targets=d
    writer.add_images("input",imgs,step)
    output=data(imgs)
    writer.add_images("output",output,step)
    step+=1

writer.close()
# 结果是output的像素比input更少，更加模糊，但是实现了减少数据（即马赛克化）



