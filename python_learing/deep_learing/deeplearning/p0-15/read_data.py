from torch.utils.data import Dataset
# help(Dataset)
import cv2
from PIL import Image
import os
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        #获得对应的地址
        # root_dir 根目录地址
        # label_path 标签地址，通常只要一个标签的名字，接下来会将他们合起来
        self.path=os.path.join(root_dir,label_dir)
    #     接下来获取对应目录下的图片地址
    #     listdir（）将目录里面的文件转化为一个列表
        self.image_path=os.listdir(self.path)

    def __getitem__(self, index):
        # 函数作用：获取其中每一个图片
        img_name=self.image_path[index]
        '''将路径的路径的相对地址，标签的名字，图片的名字拼接起来，形成一个真正的图片路径'''
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        # 获取了对应index的图片
        img=Image.open(img_item_path)
        label=self.label_dir
        return  img,label

    def __len__(self):
        return len(self.image_path)

root_dir= '../dataResourse/practise/train'
ants_label_dir='ants_image'
bees_label_dir='bees_image'
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)

img,label=ants_dataset[0]
"把两个小的数据集进行拼接"
# 作用：可以仿照一个数据集和原来的混着用
train=ants_dataset+bees_dataset

# img2,label2=train[123]
# img2.show()
