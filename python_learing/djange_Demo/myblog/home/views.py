import os

from django.shortcuts import render

# Create your views here.
from myblog.settings import BASE_DIR


def index(request, ):
    return render(request,"home/index.html")
    # template_name 模板名字

