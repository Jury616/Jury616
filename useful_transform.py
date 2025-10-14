from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer=SummaryWriter("logs")
img=Image.open("train/bees/16838648_415acd9e3f.jpg")
print(img)

# ToTensor
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    # 两个参数分别为平均值和标准差,
    # 有input[channel]=(input[channel]-mean[channel])/std[channel],
    # 对应这里input=2*input-1
img_norm=trans_norm(img_tensor)
writer.add_image("normalize",img_norm,1)
print(img_norm[0][0][0])

# Resize
print(img.size)
trans_resize=transforms.Resize((512,512))
# 生成自定义的resize工具
img_resize=trans_resize(img)    # 这里是PIL数据类型
img_resize=trans_totensor(img_resize)   # 转变为tensor数据类型
print(img_resize)
writer.add_image("Resize",img_resize,0)

# Compose -resize -第二种用法
trans_resize_2 = transforms.Resize(512)
    #  Resize(x)只有一个参数x的时候是将图片短边缩放至x，长宽比保持不变
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
# Compose()中第一个参数是一个列表，但是列表数据必须是transform型
# Compose()的作用就是将多个transform功能合并执行，防止代码冗余
img_resize_2=trans_compose(img)
writer.add_image("Resize2",img_resize_2,1)

# RandomCrop(随机裁剪)
trans_random=transforms.RandomCrop(300)
# RandomCrop(x)在x范围内对图片进行随机的裁剪，长宽比保持不变，如果用二维元组对应长宽进行随机裁剪
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()
