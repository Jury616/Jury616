from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir  # 文件夹根目录
        self.label_dir=label_dir    # label的目录
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)     # 图片的路径，并且打包为list形式


    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)


root_dir="C:\\Users\\zr\\PycharmProjects\\deep_learning\\train"
ants_label_dir="ants"
ants_dataset=MyData(root_dir,ants_label_dir)

