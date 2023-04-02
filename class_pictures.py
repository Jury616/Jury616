import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import d2l

d2l.use_svg_display()

# 通过ToTensor实例将图片从PIL类型变化成32位浮点数格式
# 并通过除以255使得所有的像素的数值均在0~1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../Data",
                                                train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../Data",
                                               train=False, transform=trans, download=True)


def get_fashion_mnist_labels(labels):
    """返回Fashion_MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show(block=True)
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))


# show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
# 这个必须在Python控制台里面运行才能生成图像


def get_dataloader_workers():
    """使用4个进程读取数据"""
    return 4


# 读取小批量的数据
batch_size = 256

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.5f} seconds')


# 综合所有的组件
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../Data", train=True,
                                                    transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../Data", train=False,
                                                   transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
num_inputs = 784
num_outputs = 10
# 输出要与类别(10个)一样多，所以权重是784*10,偏置是1*10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
