import random
import matplotlib.pyplot as plt
import numpy as np

# 生产训练数据集
n = 100                # 数据集元素个数
train_step = 1000       # 训练次数
learn_rate = 0.00005     # 学习效率
w = 2
b = 10
# y = 2x + 10
x_data = []
y_data = []
for i in range(n):
    x_data.append(i)    # x是顺序的从1到1000
    noise = random.randint(-20, 20)   # 生成一个大小在0-20之间的噪音
    y_data.append(x_data[i] * w + b + noise)    # 生成训练集中的y参数

# 训练线性回归模型
w_ = 0     # 初始化权重值
b_ = 0

for i in range(train_step):     # 训练权重值（重复train_step次）
    temp_w = 0
    temp_b = 0  
    for j in range(n):  #每次训练有n个数据
        y = x_data[j] * w_ + b_  # 计算出y的实际值

        # Loss Function 
        loss = y - y_data[j]     

        temp_w += x_data[j] * loss   
        temp_b += loss
    
    w_ -= temp_w / n * learn_rate           # 更新权重值
    b_ -= temp_b / n * learn_rate * 10000

    if i % (train_step/10) == 0:    #分成十份打印实际权重值
        print(w_)
        print(b_)


# 测试回归模型正确度（图形化显示）
plt.scatter(x_data, y_data) # 打印训练集点数据
for i in range(n):
    y_data[i] = x_data[i] * w_ + b_
plt.plot(x_data, y_data,'r') # 打印实际线性回归值
plt.show()