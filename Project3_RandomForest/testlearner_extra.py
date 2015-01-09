import numpy as np
import sys, csv, math, random
import matplotlib.pyplot as plt
import KNNLearner
import RandomForestLearner
import RandomForestLearnerBoost

reader = (csv.reader(open(sys.argv[1],'rU'),delimiter=','))
data = list(reader)
rows = len(data)
cols = len(data[1])-1

Xtrain = np.zeros(((0.6*rows),cols))
Ytrain = np.zeros((0.6*rows,1))
Xtest = np.zeros(((0.4*rows),cols))
Ytest = np.zeros((0.4*rows,1))

test_cnt = 0
train_cnt = 0
cnt = 0
X = np.zeros((rows,cols))
Y = np.zeros((rows,1))

for row in data:
    if cnt < 0.6*rows:
        for i in range(cols):
            Xtrain[train_cnt,i] = row[i]
        Ytrain[train_cnt] = row[i+1]
        train_cnt += 1
    else:
        for i in range(cols):
            Xtest[test_cnt,i] = row[i]
        Ytest[test_cnt] = row[i+1]
        test_cnt += 1
    for i in range(cols):
        X[cnt,i] = row[i]
        Y[cnt] = row[i+1]
    cnt +=1

n = 11
rms_rf1_in = np.zeros((n))
rms_rf1_out = np.zeros((n))

rms_rf_in = np.zeros((n))
rms_rf_out = np.zeros((n))

K = np.zeros((n))
corr_rf = np.zeros((n))
corr_rf1 = np.zeros((n))
for k in range(90,101):

    K[k-90] = k

    #out of sample random forest
    learner = RandomForestLearner.RandomForestLearner(k)
    d = np.hstack([Xtrain,Ytrain])
    learner.addEvidence(d[:0.6*len(d),:])
    Y_out_rf = learner.query(Xtest)
    corr_rf[k-90] = np.corrcoef(Y_out_rf,Xtest[:,-1])[0,1]

    #out of sample random forest1
    learner = RandomForestLearnerBoost.RandomForestLearner(k)
    d = np.hstack([Xtrain,Ytrain])
    learner.addEvidence(d[:0.6*len(d),:])
    Y_out_rf1 = learner.query(Xtest)
    corr_rf1[k-90] = np.corrcoef(Y_out_rf1,Xtest[:,-1])[0,1]

plt.clf()
print len(Y_out_rf1), len(Y_out_rf)
plt.plot(K, corr_rf, K, corr_rf1)
plt.legend(['CORCOEFF RF', 'CORCOEFF RF Improved'])
plt.ylabel("CORCOEFF")
plt.xlabel("K")
plt.savefig('corcoeff_rf_rf1.pdf', format='pdf')
