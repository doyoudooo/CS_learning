yinglong = [1.1496, 1.1822, 1.2153, 1.2491, 1.2832, 1.318, 1.3534, 1.3894, 1.4262, 1.4634, 1.5015, 1.5402, 1.5799,
            1.6201, 1.6613, 1.7033, 1.7462, 1.7901, 1.835, 1.8809, 1.9279]
print(len(yinglong))
LB=[]
for a in range(len(yinglong)-1):
    LB.append(yinglong[a] / yinglong[a + 1])
print(len(LB))