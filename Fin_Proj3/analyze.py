import QSTK.qstkutil.qsdateutil as du 
import QSTK.qstkutil.tsutil as tsu 
import QSTK.qstkutil.DataAccess as da 
import datetime as dt
import numpy, sys
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1], parse_dates = [[0,1,2]],
                         header = None, index_col = [0])
data.columns = ['Value']
dt_start = data.ix[0].name
dt_end = data.ix[len(data) -1].name

print 'The final value of the portfolio using the sample file is -- str ' + str(dt_end.strftime('%Y, %m, %d, ')) + str(data.Value[len(data) - 1])
print 'Details of the Performance of the portfolio'
print 'Data Range :  ' + str(dt_start + dt.timedelta(hours = 16))  + ' to ' + str(dt_end + dt.timedelta(hours = 16))

dt_end = dt_end + dt.timedelta(days = 1)
def calc_sharpe_ratio(na_price):
    daily_ret = tsu.returnize0(na_price)
    mean = numpy.mean(daily_ret)
    stdev = numpy.std(daily_ret)
    sharpe_ratio = mean * numpy.sqrt(252) / stdev 
    return sharpe_ratio

na_price = [float(integral) for integral in (data.Value)]

c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
dt_timeofday = dt.timedelta(hours=16)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
ls_symbols = [sys.argv[2]]
ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))

bench_price = d_data['close'].values
bench_price = bench_price * (data.ix[0][0]/bench_price[0])

plt.clf()
plt.plot(ldt_timestamps, bench_price)
plt.plot(ldt_timestamps, na_price)
plt.legend(ls_symbols + ['Funds'])
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
plt.savefig('adjustedclose.pdf', format='pdf')

print 'Sharpe Ratio of Fund : ' + str(calc_sharpe_ratio(na_price))
print 'Sharpe Ratio of ' + sys.argv[2].split('\\')[-1] + ' : ' + str(calc_sharpe_ratio(bench_price))

def total_ret(isBench, daily_ret):
    if not isBench:
        daily_ret = tsu.returnize0(daily_ret)
    cum_ret = [1]
    for i in range(1, len(daily_ret)):
        cum_ret.append(cum_ret[i - 1] * (1 + daily_ret[i]))
    return cum_ret[len(cum_ret) - 1]

print 'Total Return of Fund : ' + str(total_ret(False, na_price))
print 'Total Return of ' + sys.argv[2].split('\\')[-1] + ' : ' + str(total_ret(True, bench_price))

def std_dev(isBench, daily_ret):
    if not isBench:
        daily_ret = tsu.returnize0(daily_ret)
    stddev = numpy.std(daily_ret)
    return stddev

print 'Standard Deviation of Fund :  ' + str(std_dev(False, na_price))
print 'Standard Deviation of ' + sys.argv[2].split('\\')[-1] + ' : ' + str(std_dev(True, bench_price))

def avg_ret(isBench, daily_ret):
    if not isBench:
        daily_ret = tsu.returnize0(daily_ret)
    mean = numpy.mean(daily_ret)
    return mean
print 'Average Daily Return of Fund :  ' + str(avg_ret(False, na_price))
print 'Average Daily Return of ' + sys.argv[2].split('\\')[-1] + ' : ' + str(avg_ret(True, bench_price))
