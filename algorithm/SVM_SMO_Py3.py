# -*- coding:utf-8 -*-
# Author：YJX


def loadDataSet():
    """读取数据文件，并解析为特征矩阵和标签向量"""
    dataMat = []; labelMat = []
    fr = open("..\dataset\SMO_testSet.txt")
    # fr = loadD.loadDataSets("..\dataset\Logistic_testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    print("%s \n %s " % (dataMat, labelMat))