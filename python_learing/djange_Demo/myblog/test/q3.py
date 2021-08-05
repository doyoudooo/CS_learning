import math


def fuction3(day,j,w):

    b = 2 * math.pi * (day - 1) / 365
    p = 0.006918 - 0.399912 * math.cos(b) + 0.070257 * math.sin(b) - 0.006758 * math.cos(2 * b) + 0.000907 * math.sin(
        2 * b) - 0.002697 * math.cos(3 * b) + 0.00148 * math.sin(3 * b)
    # print(p)
    jindu = j  # 当前经度
    shicha = (120 - jindu) * 4
    # 当前位置和北京时间的时差
    tan=[]
    #tan比值结果
    ftan=[]

    rfinish = []
    dfinish = []
    # # 882-942

    #结果
    LB=[]
    yinglong=[1.1496	,1.1822	,1.2153,	1.2491,	1.2832	,1.318	,1.3534,	1.3894,	1.4262,	1.4634,	1.5015,	1.5402,	1.5799	,1.6201	,1.6613,	1.7033,	1.7462	,1.7901	,1.835,	1.8809,	1.9279]

    for a in range(len(yinglong)-1):
        LB.append(yinglong[a]/yinglong[a+1])

    ##最终结果min
    minp = 0
    for i in range(882,943,3):
        # 882, 942,3
        st = i - shicha
        t = 15 * (st - 720) / 60
        # 时角
        d = w
        #    %当地纬度
        d = math.radians(d)
        t = math.radians(t)

        H = math.sin(d) * math.sin(p) + math.cos(d) * math.cos(p) * math.cos(t)

        x = math.asin(H)
        finish = math.tan(x)
        tan.append(finish)

        # % 太阳高度角的真切值
    for i in range(len(tan)-1):
        s = tan[i+1]/tan[i]
        ftan.append(s)
    for i in range(20):
        minp = minp+(LB[i]-ftan[i])*(LB[i]-ftan[i])
    return LB,ftan

if __name__ == '__main__':
    (a,b)=fuction3(295, 116.391388, 39.90722)
    print(a)
    print( )
    print(b)


