import torchvision.datasets
from torch.utils.data import DataLoader
# 扑克牌堆
from torch.utils.tensorboard import SummaryWriter

test_setdata=torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor())
# 拿在手上的扑克牌
test_loader=DataLoader(dataset=test_setdata, shuffle=False, num_workers=0, drop_last=False)

img,target=test_setdata[0]
print(img.shape)
print(target)

writer=SummaryWriter('../dataloader')
for echo in range(2):
    step=0
    for data in test_setdata:
        img,target=data
        # print(img.shape)
        # print(target)
        writer.add_image('echo:{}'.format(echo),img,step)
        step=step+1

writer.close()