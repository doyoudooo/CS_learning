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