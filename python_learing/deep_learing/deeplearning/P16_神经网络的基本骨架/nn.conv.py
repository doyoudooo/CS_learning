import torch
import torch.nn.functional as F

input=torch.tensor([[1,2,0,3,1],
                                    [0,1,2,3,1],
                                    [1,2,1,0,0,],
                                    [5,2,3,1,1],
                                    [2,1,0,1,1]])
kernel=torch.tensor([[1,2,1],
                                    [0,1,0],
                                    [2,1,0]])

print(input.shape)
print(kernel.shape)
"""
input – input tensor of shape (\text{minibatch} , \text{in\_channels} , iH , iW)(minibatch,in_channels,iH,iW)
"""
# 尺寸变换
input=torch.reshape(input,[1,1,5,5])
kernel=torch.reshape(kernel,[1,1,3,3])
print(input.shape)
print(kernel.shape)

# stride  跨步——卷缩内核的跨步。可以是单个数字或元组(sH，sW)。默认值: 1
output1=F.conv2d(input,kernel,stride=1)
print(output1)

output2=F.conv2d(input,kernel,stride=2)
print(output2)

# padding  填充 -
"""
输入两侧的隐式填充。可以是一个字符串{‘ valid’，‘ same’} ，单个数字或一个元组(padH，padW)。
默认值: 0 padding = ‘ valid’与 no padding 相同。填充 = ‘ same’pads 输入，所以输出的形状作为输入。
但是，这个模式不支持除1以外的任何步幅值。
"""

output3=torch.conv2d(input,kernel,stride=1,padding=1)
print(output3)