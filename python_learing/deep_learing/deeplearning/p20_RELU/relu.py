import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.2],
                                [-1,3]])
# 因为后面（）要用的格式不对，所以要转化鸭
print(input.shape)
input=torch.reshape(input,[-1,1,2,2])
print(input.shape)


dataset=torchvision.datasets.CIFAR10('../dataset',train=False,transform=torchvision.transforms.ToTensor(),
                                     download=False)
dataloader=DataLoader(dataset,batch_size=64)

# 搭建神经网络
class relu_cc(nn.Module):
    def __init__(self):
        super(relu_cc, self).__init__()
        self.relu1=ReLU()
    #     inplace=false,默认不替换原来的数据
        self.sigmoid=Sigmoid()

    def forward(self,input):
        output=self.sigmoid(input)
        return output

relu_cc1=relu_cc()
output=relu_cc1(input)
print(relu_cc1)

# relu对图像的处理并不明显，换一个展示
writer=SummaryWriter('../logs_nonline')
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images('input',imgs,step)
    output=relu_cc1(imgs)
    writer.add_images('output',output,step)
    step+=1

writer.close()
