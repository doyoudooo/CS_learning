# -*-coding:utf-8-*-
import torch
print(torch.cuda.is_available())
for i in range(len(dir(torch))):
    print(dir(torch)[i])
