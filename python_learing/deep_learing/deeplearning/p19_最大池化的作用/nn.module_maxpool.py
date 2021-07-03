# 1.导入数据集
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor())

# 2.加载数据集
dataloader=DataLoader(dataset,batch_size=64)

# 3.创建神经网络模型
class max_cc(nn.Module):
    def __init__(self):
        super(max_cc, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=False)
    """
    最大池化的作用，一个是提取出最明显的数据，减少数据量，加快训练的速度
    
    """
    def forward(self,input):
        output=self.maxpool1(input)
        return  output

maxcc=max_cc()

writer=SummaryWriter('../logs_max')
step=0
for data in dataloader:

    img,target=data
    #_----------------------------- add_images----------------------------------!!!!!!!!!!!!!!
    writer.add_images('input',img,step)

    out= maxcc(img)
    writer.add_images('output',out,step)
    step+=1

writer.close()