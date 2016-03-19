# Machine_Learning_Foundation

已入机器学习坑，下决心走下去。《统计学习方法》一书介绍了十种算法，不算太难，但仍需重读以仔细研究其中的推导。《机器学习实战》一书则给出了各种算法的具体实例，Python实现，适合入门者了解算法的具体应用。另在Cousera上选了两门课：斯坦福Andrew Ng的《Machine Learning》、台大林田轩的《机器学习基石》和《机器学习技法》。Andrew的课程简单，省去了很多的数学推导和证明，但很全面。林田轩的课包含很多数学证明，偏难，需细细研究。

刚刚完成《机器学习基石》的第一次作业，共20个选择题。其中，前14道考察课程的理解，后6道则需code实现。下面节选了后6道编程题，通过Python实现了，保存在MLFex1.py中。

先上题目。

Question 15
For Questions 15-20, you will play with PLA and pocket algorithm. First, we use an artificial data set to study PLA. The data set is in
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat
Each line of the data set contains one (xn,yn) with xn∈R4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.
Please initialize your algorithm with w=0 and take sign(0) as −1
Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?

A   ≥201 updates

B   51 - 200 updates

C   <10 updates

D   31 - 50 updates

E   11 - 30 updates

Question 16
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm. Run the algorithm on the data set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?

A   ≥201 updates

B   11 - 30 updates

C   51 - 200 updates

D   31 - 50 updates

E   <10 updates

Question 17
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm, while changing the update rule to be
wt+1←wt+ηyn(t)xn(t)
with η=0.5. Note that your PLA in the previous Question corresponds to η=1. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?

A   51 - 200 updates

B   <10 updates

C   31 - 50 updates

D   11 - 30 updates

E   ≥201 updates

Question 18
Next, we play with the pocket algorithm. Modify your PLA in Question 16 to visit examples purely randomly, and then add the 'pocket' steps to the algorithm. We will use
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat
as the training data set D, and
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat
as the test set for ``verifying'' the g returned by your algorithm (see lecture 4 about verifying). The sets are of the same format as the previous one.
Run the pocket algorithm with a total of 50 updates on D, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

A   0.6 - 0.8

B   <0.2

C   0.4 - 0.6

D   ≥0.8

E   0.2 - 0.4

Question 19
Modify your algorithm in Question 18 to return w50 (the PLA vector after 50 updates) instead of w^ (the pocket vector) after 50 updates. Run the modified algorithm on D, and verify the performance using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

A   <0.2

B   ≥0.8

C   0.4 - 0.6

D   0.6 - 0.8

E   0.2 - 0.4

Question 20
Modify your algorithm in Question 18 to run for 100 updates instead of 50, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

A   <0.2

B   0.2 - 0.4

C   0.6 - 0.8

D   ≥0.8

E   0.4 - 0.6


