import numpy as np
import sys, csv, math, random
import matplotlib.pyplot as plt
import KNNLearner
import RandomForestLearner

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

n = 100
rms_knn_in = np.zeros((n))
rms_knn_out = np.zeros((n))

rms_rf_in = np.zeros((n))
rms_rf_out = np.zeros((n))

K = np.zeros((n))
for k in range(1,n+1):

    K[k-1] = k

    #out of sample knn
    learner = KNNLearner.KNNLearner(k)
    learner.addEvidence(Xtrain,Ytrain)
    Y_out_knn = learner.query(Xtest)
    
    sum = 0
    for i in range(len(Y_out_knn)):
        sum += math.pow((Y_out_knn[i] - Ytest[i]),2)
    rms_knn_out[k - 1] = math.sqrt(sum/len(Y_out_knn))

    #in sample knn
    learner.addEvidence(X,Y)
    Y_in_knn = learner.query(Xtest)
    
    sum = 0
    for i in range(len(Y_in_knn)):
        sum += math.pow((Y_in_knn[i] - Ytest[i]),2)
    rms_knn_in[k - 1] = math.sqrt(sum/len(Y_in_knn))


    #in sample random forest
    learner = RandomForestLearner.RandomForestLearner(k)
    d = np.hstack([X,Y])
    learner.addEvidence(d[:0.6*len(d),:])
    Y_in_rf = learner.query(Xtest)

    sum = 0
    for i in range(len(Y_in_rf)):
        sum += math.pow((Y_in_rf[i] - Ytest[i]),2)
    rms_rf_in[k - 1] = math.sqrt(sum/len(Y_in_rf))

    #out of sample random forest
    learner = RandomForestLearner.RandomForestLearner(k)
    d = np.hstack([Xtrain,Ytrain])
    learner.addEvidence(d[:0.6*len(d),:])
    Y_out_rf = learner.query(Xtest)

    sum = 0
    for i in range(len(Y_out_rf)):
        sum += math.pow((Y_out_rf[i] - Ytest[i]),2)
    rms_rf_out[k - 1] = math.sqrt(sum/len(Y_out_rf))

plt.clf()
plt.plot(K,rms_rf_out, K, rms_rf_in)
plt.legend(['RMSE RF Out', 'RMSE RF In'])
plt.ylabel("Root Mean Square Error")
plt.xlabel("K")
plt.savefig('rmse_rf_in_out.pdf', format='pdf')

plt.clf()
plt.plot(K,rms_knn_out, K,rms_rf_out)
plt.legend(['RMSE KNN', 'RMSE RF'])
plt.ylabel("Root Mean Square Error")
plt.xlabel("K")
plt.savefig('knn_vs_rf.pdf', format='pdf')
