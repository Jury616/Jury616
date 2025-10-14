import math
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from torch.utils import data


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


class Timer:
    """记录多次运行的时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def normal(x, mu, sigma):
    """正态分布函数"""
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-1.5 / sigma ** 2 * (x - mu) ** 2)


def use_svg_display():
    """使用svg格式作图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.2, 2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes: plt.Axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # 在函数参数中默认参数类型
    """设置matplotlib的轴"""
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        plt.legend(legend)
    plt.grid()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """
The function first checks if X has only one axis. If it does, it is converted to a list.
 If Y is not provided, X is set to an empty list and Y is set to X.
 If Y has only one axis, it is converted to a list. If the length of X and Y are not equal,
  X is repeated to match the length of Y.
The function then clears the current axis and plots the Data points using the fmts parameter.
Finally, the function sets the axes labels, limits, and scales, and displays the plot.
    作用：绘制数据点
    """
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = plt.Axes if axes else plt.gca()  # 必须固定axes是Axes类

    # 如果X只有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(X):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show(block=True)  # 展示图表


"""
def f(x):
    return 3*x**2-4*x
x=np.arange(0,5,0.1)
plot(x,[f(x),2*x-3],'x','f(x)',legend=['f(x)','Tangent line(x=1)'])"""
'''以上均为作图函数'''

# 梯度（求导）
"""
x=torch.arange(4.0)
x.requires_grad_(True)
y=2*torch.dot(x,x)
y.backward()
print(x.grad)
x.grad.zero_()  # 清空梯度，否则梯度会累积
y=x.sum()
y.backward()
print(x.grad)
x.grad.zero_()
y=x*x
y.sum().backward()
print(x.grad)
"""


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))  # 下标
    # 随机读取样本
    random.shuffle(indices)  # 将indices打乱以实现随机读取
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # 按顺序取batch-size大小的张量
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """均方损失"""
    return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2


def sgd(params, lr, batch_size):  # learning_rate(lr)
    """小批量随机梯度下降(更新参数)"""
    with torch.no_grad():  # torch.no_grad() 的作用是在运行该代码块中的计算时，
        # 停止跟踪并记录对张量的操作。这样可以节省内存和计算时间，在不需要梯度计算时特别有用。
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 梯度归零


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


num_inputs = 784
num_outputs = 10
# 输出要与类别(10个)一样多，所以权重是784*10,偏置是1*10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里使用了广播机制


def net(X):
    """定义模型，使图像展平为向量"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat: torch.tensor, y: torch.tensor):
    """计算正确预测的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        # argmax()获得每行最大元素的索引
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
