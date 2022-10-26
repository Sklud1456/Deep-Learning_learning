import math
import random

import numpy as np
import pandas as pd
from tqdm import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

# 创造一个指定大小的矩阵
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    result=1.0 / (1.0 + math.exp(-x))
    return result

#sigmod函数的导数
def sigmod_derivate(x):
    return sigmoid(x) * sigmoid(1 - x)

class BPNeuralNetwork():
    input_number=0
    hidden_number=0
    output_number=0
    input_cells=[]
    hidden_cells=[]
    output_cells=[]
    weights_i2h=[[]]
    weights_h2o=[[]]

    def setup(self, ni, nh, no):
        self.input_number = ni + 1  # 因为需要多加一个偏置神经元，提供一个可控的输入修正
        self.hidden_number = nh
        self.output_number = no

        # 初始化神经元，四层
        self.input_cells = self.input_number * [1.0]
        self.hidden_cells = self.hidden_number * [1.0]
        self.output_cells = self.output_number * [1.0]
        self.predict_cell = 1.0

        # 初始化权重矩阵，三个
        self.weights_i2h = make_matrix(self.input_number, self.hidden_number)
        self.weights_h2o = make_matrix(self.hidden_number, self.output_number)
        self.weights_o2p = []
        for i in range(no):
            self.weights_o2p.append(0.0)

        # 权重矩阵随机激活
        for i in range(self.input_number):
            for h in range(self.hidden_number):
                self.weights_i2h[i][h] = random.uniform(-0.5,0.5)

        for h in range(self.hidden_number):
            for o in range(self.output_number):
                self.weights_h2o[h][o] = random.uniform(-0.5,0.5)

        for o in range(self.output_number):
                self.weights_o2p[o] = random.uniform(-0.5,0.5)

    def forward_propagation(self, inputs):

        # 激活输入层
        for i in range(self.input_number - 1):
            self.input_cells[i] = inputs[i]

        # 激活隐藏1层
        for j in range(self.hidden_number):
            total = 0.0
            for i in range(self.input_number):
                total += self.input_cells[i] * self.weights_i2h[i][j]
            self.hidden_cells[j] = sigmoid(total)

        # 激活隐藏2层
        for k in range(self.output_number):
            total = 0.0
            for j in range(self.hidden_number):
                total += self.hidden_cells[j] * self.weights_h2o[j][k]
            self.output_cells[k] = sigmoid(total)

        # 激活输出层
        result = 0.0
        for out in range(self.output_number):
            result+=float(self.output_cells[out]) * self.weights_o2p[out]
        self.predict_cell=sigmoid(result)
        return self.predict_cell

    def backward_propagate(self, case, label, learn, correct):
        # 前馈
        self.forward_propagation(case)

        # 获取输出层误差
        error = label-self.predict_cell
        o2p_deltas= sigmod_derivate(self.predict_cell)*error

        # 输出层传播给隐藏2层
        output_deltas = [0.0] * self.output_number
        for o in range(self.output_number):
            error =o2p_deltas*self.weights_o2p[o]
            #获得小delta
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error

        # 2层传给1层
        hidden_deltas = [0.0] * self.hidden_number
        for h in range(self.hidden_number):
            error = 0.0
            for o in range(self.output_number):
                error += output_deltas[o] * self.weights_h2o[h][o]
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells[h]) * error

        # 更新o2p权重
        for o in range(self.output_number):
            self.weights_o2p[o] += learn * o2p_deltas * self.output_cells[o]

        # 更新h2o权重
        for h in range(self.hidden_number):
            for o in range(self.output_number):
                # deltaWkj=学习率*小delta*yj
                self.weights_h2o[h][o] += learn * output_deltas[o] * self.hidden_cells[h]

        # 更新i2h权重
        for i in range(self.input_number):
            for h in range(self.hidden_number):
                # deltaWji=学习率*小delta*yi
                self.weights_i2h[i][h] += learn * hidden_deltas[h] * self.input_cells[i]

        # 计算误差
        error = 0.5*(label-self.predict_cell) ** 2
        return error

    # 训练
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for num in tqdm(range(limit)):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.backward_propagate(case, label, learn, correct)

    # 预测
    def predict(self,cases):
        result=[]
        for i in range(len(cases)):
            case=cases[i]
            temp=self.forward_propagation(case)
            result.append(temp)
        return result

#16个变量
key=['ATSc2','BCUTc-1h','SCH-7','SsOH','minHBa','mindssC','maxsCH3','ETA_dEpsilon_C','ETA_BetaP','ETA_BetaP_s','ETA_Eta_F_L','ETA_EtaP_B_RC','FMF','MLFER_BH','WTPT-4','WTPT-5']

data1=pd.read_excel("Molecular_Descriptor.xlsx",index_col=0)
feature=data1[key]   #自变量
data2=pd.read_excel("ADMET.xlsx",index_col=0)
target=data2['MN']    #因变量
x_train, x_test, y_train, y_test = train_test_split(feature, target, random_state=4)

print(len(key))
transfer = MinMaxScaler(feature_range=(0,1))
x_train = transfer.fit_transform(x_train[key])
x_test = transfer.fit_transform(x_test[key])

# 转为np格式
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

bp=BPNeuralNetwork()
bp.setup(16,32,16)
bp.train(x_train,y_train,limit=200)
y_predict=bp.predict(x_train)

# 输出的是浮点数，四舍五入转为整数
predict1=[]
for i in y_predict:
    if i>=0.5:
        predict1.append(1)
    else:
        predict1.append(0)
cnt=0
for i in range(len(predict1)):
    if predict1[i]==y_train[i]:
        cnt+=1
print("训练集准确率：",cnt/len(predict1))

y_predict1=bp.predict(x_test)
# 输出的是浮点数，四舍五入转为整数
predict2=[]
for i in y_predict1:
    if i>=0.5:
        predict2.append(1)
    else:
        predict2.append(0)
cnt=0
for i in range(len(predict2)):
    if predict2[i]==y_test[i]:
        cnt+=1
print("测试集准确率：",cnt/len(predict2))

