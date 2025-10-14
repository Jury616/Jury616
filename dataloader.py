import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,
                                       transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
# batch_size是一次取多少个数据
writer=SummaryWriter("dataloader")
step=0
for data in test_loader:
    imgs,targets=data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("testdata",imgs,step)
    step=step+1

writer.close()
