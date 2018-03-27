# -*- coding:utf-8 -*-
# Author：YJX

import numpy as np
import re


def loadDataSet():
    """实验样本"""
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    """创建一个词列表，包含所有文档中出现的不重复的词"""
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        """循环列表"""
        vocabSet = vocabSet | set(document)  # 创建两个集合的并集
    return list(vocabSet)


def bagOfWords2Vec(vocabList, inputSet):
    """vocabList：处理好的所有的词列表；inputSet：输入的文档"""
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("单词 %s 不在列表中!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """朴素贝叶斯分类器训练函数，trainMatrix：训练的词矩阵；trainCategory：词矩阵的特征标签"""
    numTrainDocs = len(trainMatrix)  # 训练集的文档数
    numWords = len(trainMatrix[0])  # 每个文档的词数量
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 概率P（Y=1）
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  # 标签为1 的词向量的总词数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 条件概率P（X(j)=aij|Y=1）
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classfyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """朴素贝叶斯分类器，vec2Classify：待分类的向量，函数trainNB0训练的到的三个概率"""
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 属于类别1 的概率
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(longString):
    """切分文本，去掉标点、少于2个字符的单词"""
    listOfTokens = re.split(r'\W*', longString)  # 匹配非数字、字母、下划线作为分隔符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('G:\Datas\PyProjects\MLAction-master\dataset\_naiveBayes_email\spam\%d.txt' % i).read())  # 读取并解析垃圾邮件
        """注意append和extend的区别，循环完成后，docList是一个2维列表，而fullText是一个1维列表"""
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('G:\Datas\PyProjects\MLAction-master\dataset\_naiveBayes_email\ham\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 得到包含所有词的列表
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        """从50封邮件中选择10封作为测试集，并从总邮件中删除"""
        randIndex = int(np.random.uniform(0, len(trainingSet)))  # 从[0,len(trainingSet))随机采样
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []  # 训练集的词矩阵（元素的0和1 ）
    trainClasses = []  # 训练集的标签
    for docIndex in trainingSet:
        """训练，此时 len(trainingSet)=40"""
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classfyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误 %s" % (docList[docIndex]))
    print('错误率是: %s' % (float(errorCount) / len(testSet)))
    return float(errorCount) / len(testSet)

if __name__ == '__main__':
    errorRate = []
    for i in range(0, 50):
        errorRate.append(spamTest())
    print('循环 %d 次的平均错误率是：%s' % (len(errorRate), (sum(errorRate)/len(errorRate))))