from django.contrib import admin
from .models import article,article_kinds
# Register your models here.
admin.site.register(article)
admin.site.register(article_kinds)