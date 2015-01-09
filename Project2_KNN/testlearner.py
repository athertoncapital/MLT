import numpy as np
import sys, csv, math
import matplotlib.pyplot as plt
import KNNLearner
import LinRegLearner

reader = (csv.reader(open(sys.argv[1],'rU'),delimiter=','))
data = list(reader)
rows = len(data)
cols = len(data[1])-1

Xtrain = np.zeros(((0.6*rows),cols))
Ytrain = np.zeros(0.6*rows)
Xtest = np.zeros(((0.4*rows),cols))
Ytest = np.zeros(0.4*rows)

test_cnt = 0
train_cnt = 0
cnt = 0
X = np.zeros((rows,cols))
Y = np.zeros(rows)

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

n = 50
rms_knn_in = np.zeros((n))
rms_knn_out = np.zeros((n))
corr_coef_knn_in = np.zeros((n))
corr_coef_knn_out = np.zeros((n))

rms_lr_in = 0
corr_coef_lr_in = 0
rms_lr_out = 0
corr_coef_lr_out = 0

plt.clf()
plt.plot(Xtrain)

K = np.zeros((n))
Y_best_knn = []
for k in range(1,n+1):

    K[k-1] = k
    learner = KNNLearner.KNNLearner(k)
    learner.addEvidence(Xtrain,Ytrain)
    Y_out_knn = learner.query(Xtest)
    
    sum = 0
    for i in range(len(Y_out_knn)):
        sum += math.pow((Y_out_knn[i] - Ytest[i]),2)
    rms_knn_out[k - 1] = math.sqrt(sum/len(Y_out_knn))
    corr_coef_knn_out[k - 1] = np.corrcoef(Y_out_knn,Ytest)[0,1]

    learner.addEvidence(X,Y)
    Y_in_knn = learner.query(Xtest)
    
    sum = 0
    for i in range(len(Y_in_knn)):
        sum += math.pow((Y_in_knn[i] - Ytest[i]),2)
    rms_knn_in[k - 1] = math.sqrt(sum/len(Y_in_knn))
    corr_coef_knn_in[k - 1] = np.corrcoef(Y_in_knn,Ytest)[0,1]
    
    if k == 3:
        Y_best_knn = Y_out_knn

learner = LinRegLearner.LinRegLearner()
learner.addEvidence(Xtrain, Ytrain)
Y_out_lr = learner.query(Xtest)

sum = 0
for i in range(len(Y_out_lr)):
    sum += math.pow((Y_out_lr[i] - Ytest[i]),2)
rms_lr_out = math.sqrt(sum/len(Y_out_lr))
corr_coef_lr_out = np.corrcoef(Y_out_lr,Ytest)[0,1]

learner.addEvidence(X, Y)
Y_in_lr = learner.query(Xtest)

sum = 0
for i in range(len(Y_in_lr)):
    sum += math.pow((Y_in_lr[i] - Ytest[i]),2)
rms_lr_in = math.sqrt(sum/len(Y_in_lr))
corr_coef_lr_in = np.corrcoef(Y_in_lr,Ytest)[0,1]

print "In Sample - RMS Error for Linear Regression - " + str(rms_lr_in)
print "In Sample - Correlation Coefficient for Linear Regression - " + str(corr_coef_lr_in)
print "Out Sample - RMS Error for Linear Regression - " + str(rms_lr_out)
print "Out Sample - Correlation Coefficient for Linear Regression - " + str(corr_coef_lr_out)
plt.clf()
plt.plot(K,rms_knn_out, K, rms_knn_in)
plt.legend(['RMSE KNN Out', 'RMSE KNN In'])
plt.ylabel("Root Mean Square Error")
plt.xlabel("K")
plt.savefig('rmse_in_out.pdf', format='pdf')

plt.clf()
plt.scatter(Ytest, Y_best_knn)
plt.legend(['Predicted Y'])
plt.ylabel("Predicted Y")
plt.xlabel("Actual Y")
plt.savefig('predicted_vs_actual_knn.pdf', format='pdf')

plt.clf()
plt.scatter(Y_best_knn ,Y_out_lr)
plt.legend(['Predicted Y'])
plt.ylabel("Predicted Y")
plt.xlabel("Actual Y(k=11)")
plt.savefig('predicted_vs_actual_lr.pdf', format='pdf')
