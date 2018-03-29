# -*- coding:utf-8 -*-
# Author：YJX

import numpy as np
import operator
from sklearn.model_selection import train_test_split
import sys
import dataset.loadDataSets as loadD
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """K-近邻算法，四个参数分别为：输入向量、训练集、标签向量、最近的邻居数目；使用欧式距离计算公式"""
    dataSetSize = dataSet.shape[0]  #shape[0]--查看矩阵的行数，shape[1]--查看矩阵的列数
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet  # tile函数的作用：按要求重复数组inX
    # print('矩阵diffMat1：%s' % np.tile(inX, (dataSetSize, 1)))
    # print('矩阵diffMat：%s' % diffMat)
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 平方和
    distances = sqDistances**0.5  # 开方
    sortedDistIndicies = distances.argsort()  # argsort函数返回的是数组值从小到大的索引值，一个list
    classCount = {}  # 空字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get--查找字典的键voteIlabel，没有则返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 自定义排序，并按降序排列
    # print(sortedClassCount)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """读取txt文件，并解析"""
    fr = loadD.loadDataSets(filename)
    arrayOLines = fr.readlines()  #读取所有行(直到结束符 EOF)并返回列表
    numberOfLines = len(arrayOLines)  # 返回对象（字符、列表、元组等）长度或项目个数，即获取文件行数
    returnMat = np.zeros((numberOfLines, 3))  # 创建零矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 移除字符串头尾指定的字符（默认为空格），即截取所有回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 取每行数据的前三个特征
        classLabelVector.append(int(listFromLine[-1]))  # 取每行数据的最后一个特征，即标签；int(listFromLine[-1])--元素值为整型
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """数据归一化函数"""
    minVals = dataSet.min(0)  # 取每列的最小值
    maxVals = dataSet.max(0)  # 取每列的最大值
    ranges = maxVals - minVals  # 归一化公式的分母
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]  # shape函数获取dataSet的结构（即行列数），并返回一个列表；shape[0]即取返回的列表的第一个数，此处是dataSet的行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))  # normDataSet是归一化后的数据集
    return normDataSet, ranges, minVals

def datingClassTest():
    """kNN准确率测试函数,此处测试集固定：从第一行到第numTestVecs行；余下为训练集"""
    hoRatio = 0.10  # 测试集所占比例
    datingDataMat, datingLabels = file2matrix('kNN_datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    print('参数numTestVecs:%d' % numTestVecs)
    print('训练集:')
    print(normMat[numTestVecs:m, :])
    print(normMat[numTestVecs:m, :].shape[0])
    errorCount = 0.0
    for i in range(numTestVecs):
        """normMat[i, :]--数组的第i行；normMat[numTestVecs:m, :]--数组的第numTestVecs行到第m行；k=3"""
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('输入向量%d:%s' % (i, normMat[i, :]))
        # print("预测结果是： %d, 真实值是： %d"%(classifierResult, datingLabels[i]))
        if (classifierResult !=datingLabels[i]):errorCount += 1.0
    print("总错误率是： %f %%" % (errorCount*100/float(numTestVecs)))

def datingClassTestRandom(hoRatio, k):
    """kNN准确率测试函数,随机划分训练集和数据集,hoRatio 测试集所占比例"""
    datingDataMat, datingLabels = file2matrix('..\dataset\kNN_datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    normMat_train, normMat_test, Labels_train, Labels_test = train_test_split(normMat, datingLabels, train_size=1-hoRatio)
    # print('训练集行数:%s' % normMat_train.shape[0])
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat_test[i], normMat_train, Labels_train, k)
        # print('输入向量%d:%s' % (i, normMat_test[i]))
        # print("预测结果是： %d, 真实值是： %d"%(classifierResult, datingLabels[i]))
        if (classifierResult !=Labels_test[i]):errorCount += 1.0
    # print("总错误率是： %f %%" % (errorCount*100/float(numTestVecs)))
    return (errorCount*100/float(numTestVecs))

def saveToTxt(filePath, k, hoRatio, totalErrorRate):
    """save results to txt file"""
    f = open(filePath, 'a')
    try:
        strText = str(k) + '\t' + str(hoRatio) + '\t' + str(totalErrorRate) + '\n'
        f.write(strText)
    finally:
        f.close()

if __name__ == '__main__':
    # datingClassTest()
    filePath = "..\\results\\不同的k时的错误率.txt"
    f = open(filePath, 'w')
    # f.writelines('k' + '\t' + 'hoRatio' + '\t' + 'totalErrorRate' + '\n')
    f.close()
    for k in range(1, 21):
        """k取1到20，循环"""
        # hoRatio = hoRatio/100.0
        totalErrorRate = 0.0
        for i in range(10):
            totalErrorRate += datingClassTestRandom(0.10, k)
        print("k=%d,平均总错误率是： %f %%" % (k, (totalErrorRate/10)))
        saveToTxt(filePath, 0.10, k, totalErrorRate/10)
