from objective_fun_a import func_base
# 第三问附件二误差分析
l1 = func_base(220.671, 78.673, 36.597, 1.8318) #测试影长
# print(len(l1))
l2 = [1.24725620500000, 1.22279459000000, 1.19892148600000, 1.17542896400000, 1.15243957300000, 1.12991747000000,
      1.10783548000000, 1.08625420600000, 1.06508107200000, 1.04444626500000, 1.02426412600000, 1.00464031400000,
      0.985490908000000, 0.966790494000000, 0.948584735000000, 0.930927881000000, 0.913751750000000, 0.897109051000000,
      0.880973762000000, 0.865492259000000, 0.850504468000000]  #实际影长
e = 0 #北半球误差
for i in range(21):
    e = e + (abs(l1[i] - l2[i])) / l2[i]
print(e/21)

l1_south=func_base(333.552,79.887,-40.390,2.027)
e2=0  #南半球误差
for i in range(21):
    e2=e2+(abs(l1_south[i]-l2[i]))/l2[i]
print(e2/21)