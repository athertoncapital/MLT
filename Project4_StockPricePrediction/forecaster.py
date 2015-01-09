import numpy as np
from scipy.fftpack import fft,fftfreq
import scipy,math
from scipy.signal import find_peaks_cwt
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

start_date = dt.datetime(2001, 1, 1)
end_date = dt.datetime(2005,12,31)
symbols = []
for i in range(200):
    symbols += ['ML4T-'+str("%03d"%i)]
c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['close']
dt_timeofday = dt.timedelta(hours=16)
ldt_timestamps = du.getNYSEdays(start_date, end_date, dt_timeofday)
ldf_data = c_dataobj.get_data(ldt_timestamps, symbols, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))['close']

Xtest = []
Ytest = []
for sym in symbols:
    rets = tsu.returnize0(d_data[sym])[1:].ravel()
    num_days = len(rets)
    for i in range(100,num_days-4):
        st = i-100
        yf = fft(rets[st:i])
        xf = np.array(fftfreq(yf.size))*100
        N = len(yf)
        f_max = np.argmax(np.abs(yf[0:N/2]))
 
        e =  yf[f_max]/100
        phase = (np.pi/2+np.angle(e))
        amp = 2*abs(e)

        features = [phase,amp,xf[f_max]]
        val = rets[i+4]
        Xtest.append(features)
        Ytest.append(val)

Xtest = np.array(Xtest)
mean = [0,0,0]
std = [0,0,0]
for i in range(3):
    mean[i] = Xtest[:,i].mean()
    std[i] = Xtest[:,i].std()
    if Xtest[:,i].std() != 0:
        Xtest[:,i] = (Xtest[:,i]-mean[i])/std[i]
    if i == 1:
        Xtest[:,i] = abs(Xtest[:,i]*5)
learner = KNeighborsClassifier(n_neighbors=50)
learner.fit(Xtest,Ytest)

symbols = ['ML4T-292','ML4T-312']
start_date = dt.datetime(2006, 1, 1)
end_date = dt.datetime(2007,12,31)
ldt_timestamps = du.getNYSEdays(start_date, end_date, dt_timeofday)
ldf_data = c_dataobj.get_data(ldt_timestamps, symbols, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))['close']

for sym in symbols:
    Xtest = []
    Ytest = []
    rets =tsu.returnize0(d_data[sym])[1:].ravel()

    num_days = len(rets)
    for i in range(100,num_days-4):
        st = i-100
        
        yf = fft(rets[st:i])
        xf = np.array(fftfreq(yf.size))*100
        N = len(yf)
        f_max = np.argmax(np.abs(yf[0:N/2]))
        
        e =  yf[f_max]/100
        phase = (np.pi/2+np.angle(e))
        amp = 2*abs(e)
        
        features = [phase,amp,xf[f_max]]
        val = rets[i+4]
        Xtest.append(features)
        Ytest.append(val)
    
    Xtest = np.array(Xtest)
    for i in range(3):
        if Xtest[:,i].std() != 0:
            Xtest[:,i] = (Xtest[:,i]-mean[i])/std[i]
        if i == 1:
            Xtest[:,i] = abs((Xtest[:,i])*5)
    Ypredict = learner.predict(Xtest)
    
    Ypredict_1 = np.append(np.zeros(104),Ypredict[:96])
    Yactual_1 = rets[:200]
    plt.clf()
    plt.plot(range(1,201),Yactual_1,color='blue')
    plt.plot(range(1,201),Ypredict_1,color='red')
    plt.legend(['Actual','Predicted'])
    plt.ylabel('Daily Return')
    plt.xlabel('Days')
    plt.title('ActualvsPedictedReturns')
    plt.savefig(sym+'_first200returns.pdf',format='pdf')

    Ypredict_2 = Ypredict[-200:]
    Yactual_2 = rets[-200:]
    plt.clf()
    plt.plot(range(len(rets)-200,len(rets)),Yactual_2,color='blue')
    plt.plot(range(len(rets)-200,len(rets)),Ypredict_2,color='red')
    plt.legend(['Actual','Predicted'])
    plt.ylabel('Daily Return')
    plt.xlabel('Days')
    plt.title('ActualvsPedictedReturns')
    plt.savefig(sym+'_last200returns.pdf',format='pdf')

    Ypredict = np.append(np.zeros(104),Ypredict) 
    plt.clf()
    plt.scatter(rets,Ypredict)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('PredictedvsActualReturns')
    plt.savefig(sym+'_scatter.pdf',format='pdf')

    if '292' in sym:
        feature_array = Xtest[-200:]
        plt.clf()
        plt.plot(range(1,201),feature_array[:,0])
        plt.plot(range(1,201),feature_array[:,1])
        plt.plot(range(1,201),feature_array[:,2])
        plt.legend(['Phase','Amplitude','Frequency'])
        plt.ylabel('Value')
        plt.xlabel('Days')
        plt.title('Features')
        plt.savefig(sym+'_features.pdf',format='pdf')

    print sym + ": "
    print "Correlation: ", np.corrcoef(Ypredict[104:],Ytest,rowvar=0)[0][1]
    diff = (Ypredict[104:]-Ytest)
    print "RMS: " + str(np.sqrt(np.dot(diff,diff)/len(diff)))
