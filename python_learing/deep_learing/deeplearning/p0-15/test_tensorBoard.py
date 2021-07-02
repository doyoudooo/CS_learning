from torch.utils.tensorboard import SummaryWriter
import numpy
from PIL import Image
# test1
'''
writter=SummaryWriter('logs')
for i in range(100):
    writter.add_scalar('y=x',i,i)

writter.close()
'''

# test2
image_path= '../dataResourse/practise/train/ants_image/49375974_e28ba6f17e.jpg'
image_PIL=Image.open(image_path)
image_array=numpy.array(image_PIL)

# print(image_array.shape)

writter=SummaryWriter('../logs')
writter.add_image("test",image_array,2,dataformats='HWC')

writter.close()