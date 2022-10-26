import random
from random import sample
import matplotlib.pyplot as plt

def point_margin(x1,x2,w,b):#计算间隔margin
    margin = (w*x1-x2+b)*(w*x1-x2+b)/(w*w+1)
    return margin

def loss_function(real_data_x1,raw_data_x2,tag_y,w,b):#设置损失函数，这里用的是平方损失函数
    judge = raw_data_x2 -(w*real_data_x1 + b)
    if tag_y * judge > 0:
        loss = 0
    else:
        loss = (tag_y * judge) * (tag_y * judge)
    return loss

w = 2
pointnum = 1000
x1_real_data = []
x2_real_data = []
x2_raw_data = []
y_data = []
for i in range(0,pointnum):   #制作真值数据，用的是x2 = w*x1
    x1_real_data.append(i)
    x2_real_data.append(w * i)

for i in range(0,pointnum):   #增加噪声
    rawdata = x2_real_data[i] + random.gauss(0,10) + sample([40.0, -40.0], 1)[0] #增加了高斯噪声以及一个偏移量，用于支持向量机挑选最佳分类超平面
    x2_raw_data.append(rawdata)
    if rawdata > i+i: #制作标签
        y_data.append(1)
    else:
        y_data.append(-1)

# for i in range(0,100):
#     print(x1_real_data[i],x2_raw_data[i],y_data[i])

# losssum = 0
# for j in range(0,100):
#     losssum = losssum + loss_function(x1_real_data[j],y_data[j],w,1)
#     print(losssum)
bmin = -30
bmax = 31

good_b = []
for i in range(bmin,bmax):
    print('b = ',i)
    losssum = 0
    for j in range(0,pointnum):
        losssum = losssum + loss_function(x1_real_data[j],x2_raw_data[j],y_data[j],w,i)
    if losssum ==0:
        good_b.append(i)  #记录损失函数为0的b值
    print('loss = ',losssum)

print(good_b)

plt.xlim(xmax=225,xmin=-50)
plt.ylim(ymax=250,ymin=-50)
plt.plot(x1_real_data,x2_raw_data,'ro')
for i in range(0,pointnum):
    plt.annotate(y_data[i], xy = (x1_real_data[i], x2_raw_data[i]), xytext = (x1_real_data[i]+0.1, x2_raw_data[i]+0.1)) # 将标签显示在散点上



for i in range(bmin,bmax,10):
    x1 = [-100,250]
    x2 = [x1[0] + x1[0] + i,x1[1] + x1[1] + i]
    plt.plot(x1,x2)   #显示一些辅助线便于看出SVM的意义
plt.show()

min_margin = [] #记录每一个b值对应的最小距离
for i in range(0,len(good_b)):
    margin = 1000
    b = good_b[i]
    for j in range(0,pointnum):
        this_margin = point_margin(x1_real_data[j],x2_raw_data[j],w,b)
        if this_margin < margin:
            margin = this_margin
    min_margin.append(margin)

print(min_margin)

plt.plot(good_b,min_margin,'ro') #画出b值与最小距离对应的散点图
plt.show()
