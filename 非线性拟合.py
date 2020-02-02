import random
import matplotlib.pyplot as plt
import numpy as np

# 生产训练数据集
# y = x^2 + 10x + 100
a = 1
b = 10
c = 100

n = 100                   # 数据集元素个数
train_step = 10000        # 训练次数
learn_rate = 0.000005     # 学习效率
x_data = []
y_data = []
for i in range(n):
    x_data.append(i)
    noise = random.randint(0, 10)
    y_data.append(a*x_data[i]**2 + b*x_data[i] + c + noise*x_data[i])

# 训练线性回归模型
a_ = 0
b_ = 0
c_ = 0

for i in range(train_step):
    temp_w = 0
    temp_b = 0
    for j in range(n):
        y = a_*x_data[j]**2 + b_*x_data[j] + c_
        loss = y - y_data[j]
        temp_w += x_data[j] * loss
        temp_b += loss
    
    a_ -= temp_w / n * learn_rate
    b_ -= temp_b / n * learn_rate * 100
    c_ -= temp_b / n * learn_rate * 700

    if i % 500 == 0:
        print(a_)
        print(b_)
        print(c_)
        print(" ")

# 测试回归模型正确度（图形化显示）
plt.scatter(x_data, y_data)
for i in range(n):
    y_data[i] = a_*x_data[i]**2 + b_*x_data[i] + c_
plt.plot(x_data, y_data,'r')
plt.show()