import numpy as np
import matplotlib.pyplot as plt

# 设置k和x0的值
k = 0.2
x0 = 0

# 计算sigmod函数
def sigmoid(x):
    return 1 / (1 + np.exp(-k * (x - x0)))

# 生成x的值
x = np.linspace(-100, 100, 100)

# 计算y的值
y = sigmoid(x)

# 绘制sigmod函数曲线
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid Function')
plt.show()