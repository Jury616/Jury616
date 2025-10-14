import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True,
                                         transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True,
                                        transform=dataset_transform)
# torch.dataset中有很多训练好的数据集，可以直接使用，训练集就需要train=True,
# 每个数据集元素都是元组，包含img和package数据，分别为图片和图片对应的内容
# img, package = test_set[0]
# img.show()
writer=SummaryWriter("dataset")
for i in range(10):
    img,package=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()