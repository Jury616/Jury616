from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# transforms可以通过ToTensor，resize等工具将图片变成我们所需要的类型

img_path="train/ants/5650366_e22b7e1065.jpg"
img=Image.open(img_path)
# transforms的使用
tensor_trans=transforms.ToTensor()  # 自己创造一个工具
tensor_img=tensor_trans(img)

#显示图片
writer=SummaryWriter("logs")
writer.add_image("Tensor_img",tensor_img)
writer.close()

