# 说明文档

------

已入机器学习坑，下决心走下去。《统计学习方法》一书介绍了十种算法，不算太难，但仍需重读以仔细研究其中的推导。《机器学习实战》一书则给出了各种算法的具体实例，Python实现，适合入门者了解算法的具体应用。另在Cousera上选了两门课：斯坦福Andrew Ng的《Machine Learning》、台大林田轩的《机器学习基石》和《机器学习技法》。Andrew的课程简单，省去了很多的数学推导和证明，但很全面，对机器学习中的算法作了很多总结与比较。林田轩的课包含很多数学证明，偏难，需细细研究。
本repository主要记录了学习《机器学习基石》过程中的笔记和课后编程作业。

------

## 作业一

------

刚刚完成了《机器学习基石》的第一次作业，共20个选择题：其中，前14道考察课程的理解，后6道则需code实现。下面节选了后6道编程题，通过Python实现了，保存在MLFex1.py中。

### 题目

Question 15
For Questions 15-20, you will play with PLA and pocket algorithm. First, we use an artificial data set to study PLA. The data set is in

[https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat]()

Each line of the data set contains one (xn,yn) with xn∈R4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.
Please initialize your algorithm with w=0 and take sign(0) as −1
Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?
≥201 updates
51 - 200 updates
<10 updates
31 - 50 updates
11 - 30 updates

Question 16
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm. Run the algorithm on the data set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?
≥201 updates
11 - 30 updates
51 - 200 updates
31 - 50 updates
<10 updates

Question 17
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm, while changing the update rule to be
$$W_{t+1}←W_t+ηy_{n(t)}X_{n(t)}$$

with η=0.5. Note that your PLA in the previous Question corresponds to η=1. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?
51 - 200 updates
<10 updates
31 - 50 updates
D   11 - 30 updates
E   ≥201 updates

Question 18
Next, we play with the pocket algorithm. Modify your PLA in Question 16 to visit examples purely randomly, and then add the 'pocket' steps to the algorithm. We will use

[https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat]()

as the training data set D, and

[https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat]()

as the test set for ''verifying'' the g returned by your algorithm (see lecture 4 about verifying). The sets are of the same format as the previous one.
Run the pocket algorithm with a total of 50 updates on D, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?
0.6 - 0.8
<0.2
0.4 - 0.6
≥0.8
0.2 - 0.4

Question 19
Modify your algorithm in Question 18 to return w50 (the PLA vector after 50 updates) instead of w^ (the pocket vector) after 50 updates. Run the modified algorithm on D, and verify the performance using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?
<0.2
≥0.8
0.4 - 0.6
0.6 - 0.8
0.2 - 0.4

Question 20
Modify your algorithm in Question 18 to run for 100 updates instead of 50, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?
<0.2
0.2 - 0.4
0.6 - 0.8
≥0.8
0.4 - 0.6

### 代码说明

```python
# 将数据从网上down下来，存储到当前工作目录下
# 针对此exercise，最终可能保存在当前工作目录下的文件可能有三个：MLFex1_15_train.dat、MLFex1_18_train.dat、MLFex1_18_test.dat
def getRawDataSet(url):
    dataSet = urllib2.urlopen(url)
    filename = 'MLFex1_' + url.split('_')[1] + '_' + url.split('_')[2]
    with open(filename, 'w') as fr:
        fr.write(dataSet.read())
    return filename
```
```python
# 从本地文件读取训练数据或测试数据，保存X,y两个变量中
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
```
```python
# sigmoid函数，返回函数值
def sign(x, w):
    if np.dot(x, w)[0] >= 0:
        return 1
    else:
        return -1
```
```python
# 最原始的PLA训练算法
# X, Y，存储训练数据的矩阵，shape分别是(n+1)*m, m*1
# w，最初的系数矩阵，shape是(n+1)*1
# eta，参数
# updates，迭代次数
# 函数返回一个标志位（flag，用以说明训练是否结束，即最终得到的w是否完全fit训练数据），训练结果w， 实际迭代次数iterations
# 具体执行过程请阅读函数
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
```
```python
# 改进的PLA训练算法，具体改动请结合题目阅读函数
# 参数设置及返回同上一个函数
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
```
```python
# 使用背包策略训练w的PLA算法
# 参数设置同上
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
```
```python
# 针对19题使用的训练算法，结构同前面几个函数
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
```
```python
# 测试函数，返回w的错误率
def errorTest(X, y, w):
    error = 0.0   # remmember the number of the data the hypothesis doesn't fit
    rand_sort = range(len(X))
    rand_sort = random.sample(rand_sort, len(X))
    for i in range(len(X)):
        if sign(X[rand_sort[i]], w) != y[rand_sort[i], 0]:
            error += 1.0
    return error/len(X)
```
```python
# 使用此函数可直接解答15题
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
```
```python
# 使用此函数可直接解答16题
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
```




作者 [@Marcovaldo]     
2016 年 03月 24日    


