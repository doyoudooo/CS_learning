# pyTorch深度学习快速入门

## P6、Dataset类代码实战

**目标一：读取目标路径下的文件名？**

```python
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
```

![image-20210630183925774](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630183925774.png)



==主要讲解了如何读取有许多数据的文件夹里面的数据==

```python
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

root_dir='dataResourse/practise/train'
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
```



## P7、TensorBoard的使用

### 7.1、

TensorBoard 中有一个SummaryWriter函数可以画图 >-<

```python
from torch.utils.tensorboard import SummaryWriter

writter=SummaryWriter('logs')
for i in range(100):
    writter.add_scalar('y=x',i,i)

writter.close()
```





> tensorboard --logdir=logs



![image-20210630193702033](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630193702033.png)





### 读图片



#### 1.用PIL的Image度

![image-20210630200157044](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630200157044.png)

读出来的是PIL.JPEG不符合writter.add_image()的类型

![image-20210630200310807](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630200310807.png)



#### 2.用OpenCV读取图片，获得numpy类型的图片



#### 3.用numpy



![image-20210630200621624](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630200621624.png)



![image-20210630200627993](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630200627993.png)



直接运行报错

```
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
image_path='dataResourse/practise/train/ants_image/0013035.jpg'
image_PIL=Image.open(image_path)
image_array=numpy.array(image_PIL)

writter=SummaryWriter('logs')
writter.add_image("test",image_array,1)

writter.close()
```



![image-20210630201305094](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630201305094.png)



经过排查发现是sharp属性不对

默认是

![image-20210630201429477](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210630201429477.png)

而使用的image_array则是：

(512, 768, 3)与其相反

所以修改格式即可

```
dataformats='HWC'
```

```
writter.add_image("test",image_array,1,dataformats='HWC')
```