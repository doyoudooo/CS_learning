import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法-》   tensor数据类型
# 通过tensform.totensor去解决两个问题
# 1.transforms该如何使用

# 绝对路径 G:\workSpace\new_code\python_learing\deeplearning\dataResourse\practise\train\ants_image\20935278_9190345f6b.jpg
# 相对路径 dataResourse/practise/train/ants_image/67270775_e9fdf77e9d.jpg

img_path = '../dataResourse/practise/train/ants_image/67270775_e9fdf77e9d.jpg'
img = Image.open(img_path)

writer=SummaryWriter("../logs")
print(img)
# 实例化对象，再调用，实现call方法， 再传进去参数    transforms不能传参
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)#ctrl+p
# print(tensor_img)

writer.add_image('tensor_img',tensor_img)
writer.close()

# 2.tensor数据类型相比其他数据类型有什么区别
cv_img=cv2.imread(img_path)
#d都是ToTensor的可读取图片的两种类型之一
