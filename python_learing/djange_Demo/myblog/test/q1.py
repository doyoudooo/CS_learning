import math


def fuction1(h,j,w):
    # b太阳高度角中的中间变量
    # p 太阳赤角
    day=295
    b=2*math.pi*(day-1)/365
    p=0.006918-0.399912*math.cos(b)+0.070257*math.sin(b)-0.006758*math.cos(2*b) +0.000907*math.sin(2*b)-0.002697*math.cos(3*b)+0.00148*math.sin(3*b)
    # print(p)
    jindu=j#当前经度
    shicha=(120-jindu)*4
# 当前位置和北京时间的时差
    rfinish=[]
    dfinish=[]
    # # 882-942
    for i in range(540,900):
        st=i-shicha
    # %北京时间的9-15点在当地时间的时刻
        # 882-942
        # t=[i for i in range(60)]
        # for i in range(60):
        #     t[i] = 15 * (st[i] - 720) / 60
        t=15*(st-720)/60
        # 时角
        d=w
    #    %当地纬度
        d=math.radians(d)
        t=math.radians(t)

        H = math.sin(d) * math.sin(p) + math.cos(d) * math.cos(p) * math.cos(t)

        x = math.asin(H)
        finish=math.tan(x)

        # % 太阳高度角的真切值

        L=h/finish
        dL=-h/(finish*finish)
        rfinish.append(L)
        dfinish.append(dL)
    return rfinish,dfinish

#
a,b=fuction1(3,116.391388,39.90722,)

print(a)
print()
print(b)

