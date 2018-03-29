# -*- coding:utf-8 -*-
# Author：YJX

import numpy as np
import matplotlib.pyplot as plt
import math
import dataset.loadDataSets as loadD

def loadDataSet():
    """读取数据文件，并解析为特征矩阵和标签向量"""
    dataMat = []
    labelMat = []
    fr = open("..\dataset\Logistic_testSet.txt")
    # fr = loadD.loadDataSets("..\dataset\Logistic_testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    """Sigmoid函数，输入inX是n维行向量，对其元素依次求sigmoid函数值，并存在列表b"""
    b = []
    for i in range(len(inX)):
        a = 1/(1+math.exp(-inX[i]))
        b.append(a)
    return np.mat(b).transpose()  # 先将列表b转换为行向量，然后转置为列向量并返回

def gradAscent(dataMatIn, classLabels):
    """梯度上升法"""
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    """梯度上升法的迭代次数，迭代次数太少则权值weights未收敛，线性模型的分类效果差，如:迭代100次之分类效果远差于迭代500次的效果。怎么确定迭代次数？"""
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error
#         print("weights:%s"% weights)
    return weights

def plotBestFit(weights):
    """画数据集的散点图"""
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.2)
    # weights = weights.getA()
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def sigmoid1(inX):
    return 1/(1+math.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """随即梯度上升法，alpha不固定，仅用一个样本更新回归系数"""
    IterCount = 0
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            IterCount += 1
            # 从一个均匀分布[0,len(dataIndex))中随机采样
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid1(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            for k in range(len(dataMatrix[randIndex])):
                weights[k] = weights[k] + alpha * error * dataMatrix[randIndex][k]
            del(dataIndex[randIndex])
            saveToTxt("..\\results\\weights_of_Stochastic_grad.txt", IterCount, alpha, weights[0], weights[1], weights[2])
    return weights

def saveToTxt(filePath, IterCount, alpha, weights0, weights1, weights2):
    """save results to txt file"""
    f = open(filePath, 'a')
    try:
        strText = str(IterCount) + '\t' + str(alpha) + '\t' + str(weights0) + '\t' + str(weights1)+ '\t' + str(weights2)+ '\n'
        f.write(strText)
    finally:
        f.close()

if __name__ == '__main__':
    filePath = "..\\results\\weights_of_Stochastic_grad.txt"
    f = open(filePath, 'w')
    f.close()
    dataMatIn, classLabels = loadDataSet()
    weights = stocGradAscent1(dataMatIn, classLabels, 200)
    print(weights)
    plotBestFit(weights)