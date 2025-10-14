import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 卷积核的实现
input=torch.tensor([[1,2,0,3,1],
                   [0,1,2,3,1],
                   [1,2,1,0,0],
                   [5,2,3,1,1],
                   [2,1,0,1,1]])
kernel=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])
input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))

output1=F.conv2d(input,kernel,stride=1)
# conv2d函数是kernel在input上进行移动，最初与input的左上角重合，再分别向右向下移动stride单位，
# kernel的各个元素与input相同位置相乘后相加得到output对应位置的输出
print(output1)
output2=F.conv2d(input,kernel,stride=2)
print(output2)
output3=F.conv2d(input,kernel,padding=1,stride=1)
# 设置padding时会将input向外拓展一个单位，kernel最初就与拓展后的input左上角重合
print(output3)

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,
                                       transform=torchvision.transforms.ToTensor(),download=True)
data_loader=DataLoader(dataset,batch_size=64)

class data(nn.Module):
    def __init__(self):
        super(data,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        # in_channels=3说明是彩色图像

    def forward(self,x):
        self.conv1(x)
        return x


data_=data()
writer=SummaryWriter("l")
step=0
for d in data_loader:
    imgs,targets=d
    output=data_(imgs)
    print('imgs',imgs.shape)
    torch.reshape(output,(64,6,32,-1))
    print('output',output.shape)
    writer.add_images("input",imgs,step)
    # torch.Size([64, 3, 32, 32]
    writer.add_images("output",output,step)
    step+=1

writer.close()
