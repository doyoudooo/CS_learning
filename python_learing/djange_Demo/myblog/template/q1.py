import math


def fuction1(h,j,w):
    # b太阳高度角中的中间变量
    # p 太阳赤角
    b=2*math.pi*(108)/365
    p=0.006918-0.399912*math.cos(b)+0.070257*math.sin(b)-0.006758*math.cos(2*b) +0.000907*math.sin(2*b)-0.002697*math.cos(3*b)+0.00148*math.sin(3*b)

    jindu=j#当前经度
    shicha=(120-jindu)*4
# 当前位置和北京时间的时差

# st = [540-shicha:900-shicha]
    #st=range(882-shicha,942-shicha+1,1)

    st=[i for i in range(60)]
    st[0]=882 - shicha
    for i in range(1,60):
        st[i]= st[i-1]+1
# %北京时间的9-15点在当地时间的时刻
    # 882-942
    t=[i for i in range(60)]
    for i in range(60):
        t[i] = 15 * (st[i] - 720) / 60

    # 时角
    d=w
#    %当地纬度
    p=p*180/math.pi
    print(d)
    print(p)

# H = sind(d) * sin(p) + cosd(d) * cos(p) * cosd(t)
    H=[i for i in range(60)]
    for i in range(60):
        H[i] = math.sin(d) * math.sin(p) + math.cos(d) * math.cos(p) * math.cos(t[i])


# %太阳高度角
    x=[i for i in range(60)]
    for i in range(60):
        x[i] = math.asin(H[i])

    #     %反三角函数
    finish=[i for i in range(60)]
    for i in range(60):
        finish[i]=math.tan(x[i])
    # % 太阳高度角的真切值

    L=[i for i in range(60)]
    for i in range(60):
        L[i]=h/finish[i]
    return L

#
a=fuction1(3,39.90722,116.391388)
print(a)


