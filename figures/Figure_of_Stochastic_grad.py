# -*- coding:utf-8 -*-
# Author：YJX
import matplotlib.pyplot as plt

"""随机梯度上升法中产生的alpha和weights变化曲线"""

def loadData():
    """读取txt文件"""
    numIter = []
    weights0 = []
    weights1 = []
    weights2 = []
    alpha = []
    fr = open("../results/weights_of_Stochastic_grad.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        numIter.append(int(lineArr[0]))
        alpha.append(float(lineArr[1]))
        weights0.append(float(lineArr[2]))
        weights1.append(float(lineArr[3]))
        weights2.append(float(lineArr[4]))
    return numIter, alpha, weights0, weights1, weights2


def plotFit():
    """绘图"""
    numIter, alpha, weights0, weights1, weights2 = loadData()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.plot(numIter, alpha)
    # plt.xlabel('numIter');plt.ylabel('alpha')
    ax2.plot(numIter, weights0)
    ax3.plot(numIter, weights1)
    ax4.plot(numIter, weights2)
    plt.show()

if __name__ == '__main__':
    plotFit()