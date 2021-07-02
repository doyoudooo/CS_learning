import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
import torchvision
train_set=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=dataset_transform,download=True)

# print(test_set[0])
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes)
# print(train_set.classes[target])
# img.show()

# print(test_set[0])
writer=SummaryWriter('../p10')
for i in range(10):
    img,target=test_set[i]
    writer.add_image('dataset',img,i)

writer.close()