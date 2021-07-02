"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
"""
# 复习
# 1.先导包，torch和torchvision
import torch
import torchvision

# 2.获取训练数据集
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 3.加载数据集
dataloader=DataLoader(dataset,batch_size=64)

# 4.创建神经网络模型
class Cc(nn.Module):
    def __init__(self):
        super(Cc, self).__init__()
        self.cov1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
#         输入的是彩色图片，rgb三通道，所以in_channels=3，
    def forward(self,x):
        x=self.cov1(x)
        return x

#5. 生成一个小的神经网络
cc=Cc()
# print(cc)

#     6.tensorboard复习 ____  tensorboard --logdir=logs
writter=SummaryWriter('../logs')
step=0

for data in dataloader:
    # print(data)
    # 每一个data有两个输出，一个是img,另一个是target
    imgs, target=data
    output=cc(imgs)
    # print(img.shape)      #torch.Size([64, 3, 32, 32])     64的batch_size，in——channel=3，
    # print(output.shape)   #torch.Size([64, 6, 30, 30])   经过卷积操作之后，变成了6个channel，但是原始图像的大小变小了
    """
    writter.add_images
    """
    writter.add_images('input', imgs, step)   #torch.Size([64, 3, 32, 32])
    # torch.Size([64, 6, 30, 30])>>>   ([xx, 3, 30, 30])
    output=torch.reshape(output,[-1,3,30,30])
    writter.add_images('output',output,step)  #torch.Size([64, 6, 30, 30])
    step+=1


