from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# 写入图片
writer=SummaryWriter("logs")    # 开拓存储文件夹
image_path="train/ants/5650366_e22b7e1065.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(img_array.shape)
writer.add_image("test",img_array,1,dataformats='HWC')
# 第一个参数是标题名称，第二个是图片的array或者tensor向量，第三个是第几步，第四个是向量的形式，numpy型为HWC


# y=3*x的图像
for i in range(100):
    writer.add_scalar("y=3*x",i,i*3)
# 第一个参数是做出来的图像的名称，第二个参数是x，第三个参数是y轴

writer.close()


# 最后在终端输入tensorboard --logdir=logs (-channel=xxxx)
# 即可在本地浏览器中浏览图像和作出的图像
