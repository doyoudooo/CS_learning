import os

from django.shortcuts import render

# Create your views here.
from home.models import article, article_kinds
from myblog.settings import BASE_DIR


def index(request, ):
    return render(request,"base.html")
    # template_name 模板名字

def getArticleDetail(request,id):
    articleObject=article.objects.get(id=id)
    return render(request,'home/detail.html',locals())

def getArticleByTitle(request,):
    title=request.GET.get('title')
    print(title)
    article_list=article.objects.filter(title__icontains=title)
    return render(request,'home/serchResultList.html',locals())

def getArticleKindById(request,id):
    kind=article_kinds.objects.get(id=id)
    article_list=article.objects.filter(kind=kind)
    return  render(request,'home/kind.html',locals())

# def index(request,):
#     article_list=article.objects.all()
#     return render(request,'home/articleList',locals())





def about(request,):
    return render(request,'home/about.html')


