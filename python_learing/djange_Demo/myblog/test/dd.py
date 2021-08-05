st=[]
for i in range(100):
    st[i]=0
st = [0,1,2]
st[0] = 882 - 5
last = 942 - 5
print(st)
for i in range(1, last-1):
    st[i] = st[i-1] + 1
print(st)