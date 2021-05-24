
import numpy as np
import torch
from sklearn import preprocessing as pp
import random
import matplotlib.pyplot as plt

# （1）参数定义：
# 初始化参数：
# 初始化参数矩阵：
np.random.seed(10)  # 固定随机数种子
W1_raw = np.random.rand(10, 4)
W2_raw = np.random.rand(8, 10)
W3_raw = np.random.rand(3, 8)

global Show_gra
Show_gra = 1  # 输出梯度

# 学习率：
Alfa = 0.01   # 和这个关系不大
# 迭代次数：
Run_num = 500

# （2）文件读取：
Data = np.loadtxt('../Data/iris.data', dtype=bytes, delimiter=',').astype(str)  # 读取的数据是‘字符串’格式的
Num_all = Data.shape[0]   # 样本总的个数

# 整型编码：Iris-setosa--0; Iris-versicolor--1; Iris-virginica--2
for i in range(Num_all):
    if Data[i][4] == 'Iris-setosa':
        Data[i][4] = 0
    elif Data[i][4] == 'Iris-versicolor':
        Data[i][4] = 1
    else:
        Data[i][4] = 2

# 将字符数据转成‘float’数据：
Data = Data.astype(float)

# 对前四列特征进行‘Z-Score’标准化：
for i in range(4):
    Data[:, i] = pp.scale(Data[:, i])

# 将数据划分为训练集（80%），验证集（10%）和检验集（10%）：
Num_train = int(Num_all * 0.8)  # 训练集样本的个数
Num_valid = int(Num_all * 0.1)
Num_test = int(Num_all * 0.1)

# 将[0, 149]数组打乱顺序，选取前80%的下标对应的数据为训练集：
Index_all = list(range(Num_all))
# 固定随机数种子：
random.seed(10)
random.shuffle(Index_all)
Index_train = Index_all[0:Num_train]  # 训练集的下标
Index_valid = Index_all[Num_train:Num_train+Num_valid]
Index_test = Index_all[Num_train+Num_valid:]

# 训练集、验证集、检验集：
# 对数据集X进行转置，一个数据特征对应一个列向量：
X_train, Y_train = Data[Index_train, 0:4].T, Data[Index_train, 4]
X_valid, Y_valid = Data[Index_valid, 0:4].T, Data[Index_valid, 4]
X_test, Y_test = Data[Index_test, 0:4].T, Data[Index_test, 4]

#################################  函数定义  #################################
# 计算Loss：
def cost_fun(X, Y, W1, W2, W3):
# X：特征数据，每一行对应一个数据的特征
# Y: 标签数据
# W:参数矩阵
    Sample_num = X.shape[1]  # 样本的数量
    z1 = np.dot(W1, X)
    h1 = 1 / (1 + np.exp(-z1))  # 隐藏层1的参数

    z2 = np.dot(W2, h1)
    h2 = 1 / (1 + np.exp(-z2))

    z3 = np.dot(W3, h2)
    # 正确率：
    Precision = 0
    Right_num = 0
    # 计算交叉熵：
    J = 0  # 交叉熵初始化
    for i in range(Sample_num):
        y_z3 = z3[:, i]  # 一个预测结果
        y = np.exp(y_z3) / sum(np.exp(y_z3))  # softmax归一化
        J = J - np.log(y[int(Y[i])])
        # 最大值对应的下标：
        max_index = np.where(y == max(y))[0][0]
        if max_index == int(Y[i]):
            Right_num = Right_num + 1

    J = J / Sample_num
    Precision = Right_num / Sample_num

    # 返回损失函数和正确率：
    return J, Precision

# 返回矩阵的梯度：
def back_pro(X, Y, W1, W2, W3, Alfa):
    global Show_gra
    # 应用梯度下降法对参数进行优化，默认batch=1
    # Alfa: 学习率
    Sample_num = X.shape[1]  # 样本的数量
    # 对每个样本应用‘梯度下降’算法：
    for i in range(Sample_num):
        x = X[:, i]  # 一个样本
        x = x.reshape(x.shape[0], 1)

        # 正传：
        z1 = np.dot(W1, x)
        h1 = 1 / (1 + np.exp(-z1))  # 隐藏层1的参数
        z2 = np.dot(W2, h1)
        h2 = 1 / (1 + np.exp(-z2))
        z3 = np.dot(W3, h2)
        y = np.exp(z3) / sum(np.exp(z3))  # 是对z3进行softmax归一化！！！！

        # 计算三个矩阵的梯度：
        W1_gra = np.zeros(W1.shape)
        W2_gra = np.zeros(W2.shape)
        W3_gra = np.zeros(W3.shape)

        y_true = np.zeros((3, 1))
        y_true[int(Y[i])] = 1  # 真实的标签
        ls3 = y - y_true

        W3_gra = W3_gra + np.dot(ls3, h2.T)

        # Sigmoid的导数函数：
        s2_Deri = (1 / (1 + np.exp(-z2))) * (1 - 1 / (1 + np.exp(-z2)))
        s1_Deri = (1 / (1 + np.exp(-z1))) * (1 - 1 / (1 + np.exp(-z1)))

        W3s2 = np.dot(W3.T, ls3) * s2_Deri
        W2_gra = W2_gra + np.dot(W3s2, h1.T)
        W1_gra = W1_gra + np.dot(np.dot(W2.T, W3s2) * s1_Deri, x.T)

        # 输出手动计算的梯度：
        if Show_gra and i == 0:
            print('手动梯度W1:')
            print(torch.tensor(W1_gra))
            print('手动梯度W2:')
            print(torch.tensor(W2_gra))
            print('手动梯度W3:')
            print(torch.tensor(W3_gra))
            Show_gra = 0  # 只显示一次

        # 更新参数矩阵：
        W1 = W1 - Alfa * W1_gra
        W2 = W2 - Alfa * W2_gra
        W3 = W3 - Alfa * W3_gra
    return W1, W2, W3

# 迭代优化模型
train_loss = []
valid_loss = []
W1, W2, W3 = W1_raw, W2_raw, W3_raw
for i in range(Run_num):
    # 应用梯度下降法优化矩阵：
    [W1, W2, W3] = back_pro(X_train, Y_train, W1, W2, W3, Alfa)
    # Loss和准确度：
    [J_train, Precision_train] = cost_fun(X_train, Y_train, W1, W2, W3)
    [J_valid, Precision_valid] = cost_fun(X_valid, Y_valid, W1, W2, W3)
    [J_test, Precision_test] = cost_fun(X_test, Y_test, W1, W2, W3)
    train_loss.append(J_train)
    valid_loss.append(J_valid)

# 输出模型准确率：
print('\n  模型准确率：')
print('  训练集：{}%； 验证集：{}%； 测试集：{}%。'.format(Precision_train*100, Precision_valid*100, Precision_test*100))

# 绘图：
plt.figure()
plt.plot(train_loss, color='b', label='Train loss')
plt.plot(valid_loss, color='r', label='Valid loss')
plt.title('Loss via echo')
plt.xlabel('echo')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 使用train集中第一个样本，使用pytorch自动计算梯度：
x = X_train[:, 0]  # 特征
y = Y_train[0]
x = torch.tensor(x)
y = torch.tensor(y)
w1 = torch.tensor(W1_raw, requires_grad=True)  # 注意这里是小写的w
w2 = torch.tensor(W2_raw, requires_grad=True)
w3 = torch.tensor(W3_raw, requires_grad=True)
# 正传：
z1 = torch.mv(w1, x)
h1 = torch.sigmoid(z1)  # 隐藏层1的参数

z2 = torch.mv(w2, h1)
h2 = torch.sigmoid(z2)

z3 = torch.mv(w3, h2)
y_out = torch.exp(z3) / sum(torch.exp(z3))  # softmax归一化
loss = -torch.log(y_out[int(y)])
loss.backward()

# 输出torch自动计算的梯度：
print('自动梯度W1:')
print(w1.grad)
print('自动梯度W2:')
print(w2.grad)
print('自动梯度W3:')
print(w3.grad)