import torch
import d2l

'''
#读取和处理数据集
os.makedirs(os.path.join('..', 'Data'), exist_ok=True)
data_file = os.path.join('..', 'Data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
Data = pd.read_csv(data_file)
print(Data)
inputs, outputs = Data.iloc[:, 0:2], Data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
#使每一个NA变为该列的平均值mean
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
#将NA换成0，其他的换成1
print(inputs)'''

"""from torch.distributions import multinomial
fair_probs=torch.ones([6])/6
counts=multinomial.Multinomial(10,fair_probs).sample((500,))
cum_counts=counts.cumsum(dim=0)
estimates=cum_counts/cum_counts.sum(dim=1,keepdims=True)
d2l.set_figsize((10,6))
for i in range(6):
    d2l.plt.plot(estimates[:,i].numpy(),label=("P(die="+str(i+1)+")"))
d2l.plt.axhline(y=0.167,color='black',linestyle='dashed')
d2l.plt.xlabel('Groups of experiments')
d2l.plt.ylabel('Estimated probability')
d2l.plt.legend()    #显示直线图例
d2l.plt.show(block=True)    # 添加block=True防止窗口运行超时"""

"""
x=np.arange(-7,7,0.0001)
params=[(0,1),(0,2),(3,1)]
d2l.plot(x,Y=[normal(x,mu,sigma) for mu,sigma in params],xlabel='x',ylabel='p(x)',
         figsize=(10.0,10.0),legend=[f'mean {mu},std {sigma}' for mu,sigma in params])
"""

# 生成w=[2,-3.4].T  ,b=4.2以及噪声项的结果，并绘制散点图
true_w = torch.tensor([2, -3.4])
true_b = 4.2
batch_size = 1000
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
"""d2l.set_figsize((20,20))
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
d2l.plt.show(block=True)

for X,y in d2l.data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
"""

"""timer=d2l.Timer()
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
lr=0.03
num_epochs=3            # 迭代次数
net=d2l.linreg
loss=d2l.squared_loss
for epoch in range(num_epochs):
    for X,y in d2l.data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)    # X和 y的小批量损失
        # 因为l形状是(batch_size,1),而不是一个标量，所以要让l中所有元素被加到一起，并以此计算[w,b]的梯度
        l.sum().backward()
        d2l.sgd([w,b],lr,batch_size)    # 使用参数的梯度进行更新参数
        with torch.no_grad():
            train_l=loss(net(features,w,b),labels)
            print(f"epoch {epoch+1}, loss {float(train_l.mean()):f}")
print(f"w的估计误差:{true_w-w.reshape(true_w.shape)}")
print(f"b的估计误差:{true_b-b}")
print(f"Total time:{timer.stop():.5f} s")"""

# 读取数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
batch_size = 1000
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
data_iter = d2l.load_array((features, labels), batch_size)
# 数据很多，因此打包为迭代器
print(next(iter(data_iter)))

# 神经网络模型
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
# 单层网络架构被称为全连接层，在Linear中被定义，第一个参数指定输入特征形状，第二个指定输出特征形状
# 初始化模型参数(用normal_()和fill_()重写参数值)
net[0].weight.Data.normal_(0, 0.01)  # 权重从均值为0，标准差为0.01的正态分布中随机采样
net[0].bias.Data.fill_(0)
loss = nn.MSELoss()  # 定义L2范数的损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 小批量随机梯度下降算法实现优化参数，参数就是net.parameters()返回的参数
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 遍历数据集
        l = loss(net(X), y)
        # 调用net()生成预测，并计算损失l
        trainer.zero_grad()
        l.backward()
        trainer.step()
        # 通过反向传播计算梯度
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# w=net[0].weight.Data
# b=net[0].bias.Data
# print('w的估计误差：',true_w-w.reshape(true_w.shape))
# print('b的估计误差：',true_b-b)
