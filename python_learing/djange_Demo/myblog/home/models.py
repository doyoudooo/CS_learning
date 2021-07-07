from django.db import models

# Create your models here.
# 2.1.4
from django.db.models import Model

# 文章分类
class article_kinds(models.Model):
    kinds=models.CharField('类别',max_length=20,blank=False,null=False)

    def __str__(self):
        return  self.kinds

#  文章
class article(models.Model):
    # 标题
    title=models.CharField(max_length=50,blank=False,null=False,default='',verbose_name="标题")
    # 作者
    auther=models.CharField(max_length=50,blank=True,null=True,default='',verbose_name="作者")
    # 内容
    content=models.TextField(default='',verbose_name="正文")
    # 提交时间
    create_time=models.DateTimeField(blank=False,null=False,auto_now_add=True)
    # 分类
    kind=models.ForeignKey(article_kinds,on_delete=models.CASCADE,verbose_name='分类')
#     on_delete=models.CASCADE 不能加“”

    def __str__(self):
        return  self.title
