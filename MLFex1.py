# -*- coding: UTF-8 -*-
'''
@author Marcovaldo
@time 2016/3/18
'''
import urllib2
import numpy as np
import random
import time

# 将数据从网上down下来，存储到当前工作目录下
def getRawDataSet(url):
    dataSet = urllib2.urlopen(url)
    filename = 'MLFex1_' + url.split('_')[1] + '_' + url.split('_')[2]
    with open(filename, 'w') as fr:
        fr.write(dataSet.read())
    return filename
# 从本地文件读取数据X, y
def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()   # 将训练数据读出，存入dataSet变量中
    num = len(dataSet)  # 训练数据的组数
    # 提取X, Y
    X = np.zeros((num, 5))
    Y = np.zeros((num, 1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X[i, 0] = 1.0
        X[i, 1] = np.float(data[0])
        X[i, 2] = np.float(data[1])
        X[i, 3] = np.float(data[2])
        X[i, 4] = np.float(data[3])
        Y[i, 0] = np.int(data[4])
    return X, Y

def sign(x, w):
    if np.dot(x, w)[0] >= 0:
        return 1
    else:
        return -1

def trainPLA_Naive(X, Y, w, eta, updates):
    iterations = 0  # 记录实际迭代次数
    num = len(X)    # 训练数据的个数
    flag = True
    for i in range(updates):
        flag = True
        for j in range(num):
            if sign(X[j], w) != Y[j, 0]:
                flag = False
                w += eta * Y[j, 0] * np.matrix(X[j]).T
                break
            else:
                continue
        if flag == True:
            iterations = i
            break

    return flag, iterations, w


def trainPLA_Fixed(X, Y, w, eta, updates):
    iterations = 0  # 记录实际迭代次数
    num = len(X)    # 训练数据的个数
    flag = True
    for i in range(updates):
        flag = True
        rand_sort = range(len(X))
        rand_sort = random.sample(rand_sort, len(X))
        for j in range(num):
            if sign(X[rand_sort[j]], w) != Y[rand_sort[j], 0]:
                flag = False
                w += eta * Y[rand_sort[j], 0] * np.matrix(X[rand_sort[j]]).T
                break
            else:
                continue
        if flag == True:
            iterations = i
            break
    return flag, iterations, w

def pocketPLA(X, Y, w, eta, updates):
    num = len(X)    # 训练数据的个数
    for i in range(updates):
        rand_sort = range(len(X))
        rand_sort = random.sample(rand_sort, len(X))
        for j in range(num):
            if (sign(X[rand_sort[j]], w) != Y[rand_sort[j], 0]):
                wt = w + eta * Y[rand_sort[j], 0] * np.matrix(X[rand_sort[j]]).T
                errate0 = errorTest(X, Y, w)
                errate1 = errorTest(X, Y, wt)
                if errate1 < errate0:
                    w = wt
                break

    return w

def trainPLA(X, Y, w, eta, updates):
    num = len(X)
    for i in range(updates):
        rand_sort = range(len(X))
        rand_sort = random.sample(rand_sort, len(X))
        for j in range(num):
            if (sign(X[rand_sort[j]], w) != Y[rand_sort[j], 0]):
                w += eta * Y[rand_sort[j], 0] * np.matrix(X[rand_sort[j]]).T
                break
    return w


def errorTest(X, y, w):
    error = 0.0   # remmember the number of the data the hypothesis doesn't fit
    rand_sort = range(len(X))
    rand_sort = random.sample(rand_sort, len(X))
    for i in range(len(X)):
        if sign(X[rand_sort[i]], w) != y[rand_sort[i], 0]:
            error += 1.0
    return error/len(X)


def question15():
    url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat'
    filename = getRawDataSet(url)
    X, y = getDataSet(filename)
    w0 = np.zeros((5, 1))
    eta = 1
    updates = 80
    flag, iterations, w = trainPLA_Naive(X, y, w0, eta, updates)
    print flag
    print iterations
    print w


def question16():
    # url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat'
    filename = 'MLFex1_15_train.dat'         # getRawDataSet(url)
    X, y = getDataSet(filename)
    w0 = np.zeros((5, 1))
    eta = 0.5
    updates = 200
    times = []
    for i in range(2000):
        w0 = np.zeros((5, 1))
        flag, iterations, w = trainPLA_Fixed(X, y, w0, eta, updates)
        if flag == True:
            times.append(iterations)
    print times
    print len(times)
    return sum(times)/len(times)


# question15()

'''
# 第16题执行代码
start = time.clock()
print question16()
end = time.clock()
print 'Time consumption: %f s' % (end - start)
'''


# 第18/19/20题执行代码
# url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat'
# filename = getRawDataSet(url)   # 训练数据存储在filename文件中
start = time.clock()
filename1 = 'MLFex1_18_train.dat'
# url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat'
# filename2 = getRawDataSet(url)
filename2 = 'MLFex1_18_test.dat'
errate = []
for i in range(200):
    X, y = getDataSet(filename1)
    w = np.zeros((5, 1))
    eta = 0.5
    updates = 100
    # 使用pocket PLA训练w
    w = pocketPLA(X, y, w, eta, updates)
    X, y = getDataSet(filename2)
    print i
    print errorTest(X, y, w)
    errate.append(errorTest(X, y, w))
print sum(errate)/len(errate)
end = time.clock()
print 'Time consumption: %f s' % (end - start)
