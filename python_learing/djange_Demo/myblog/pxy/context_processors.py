from home.models import article_kinds
def articleKind(request,):
    articleKind=article_kinds.objects.all()
    context={'kind':articleKind}
    return context