import QSTK.qstkutil.qsdateutil as du 
import QSTK.qstkutil.tsutil as tsu 
import QSTK.qstkutil.DataAccess as da 
import datetime as dt
import numpy, sys, time

ls_symbols = numpy.array(sys.argv[7:])
dt_start = dt.datetime(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
dt_end = dt.datetime(int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

def simulate(dt_start, dt_end, ls_symbols, alloc):
    c_dataobj = da.DataAccess('Yahoo')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    na_price = d_data['close'].values
    na_normalized_price = na_price / na_price[0, :]
    na_normalized_price = na_normalized_price * alloc
    na_normalized_price[:, 0] = numpy.sum(na_normalized_price, 1)
        
    na_rets = na_normalized_price[:, 0].copy()
    daily_ret = tsu.returnize0(na_rets)
    mean = numpy.mean(daily_ret)
    stdev = numpy.std(daily_ret)
    sharpe_ratio = mean * numpy.sqrt(252) / stdev 
    cum_ret = [1]
    for i in range(1, len(daily_ret)):
        cum_ret.append(cum_ret[i - 1] * (1 + daily_ret[i]))
    return (stdev , mean, sharpe_ratio, cum_ret[len(cum_ret) - 1])

max_sharpe_ratio = 0
max_data = []
for a in numpy.arange(0.0, 1.0, 0.1):
    for b in numpy.arange(0.0, 1.0 - a, 0.1):
        for c in numpy.arange(0.0, 1.0 - a - b, 0.1):
            d = 1.0 - a - b - c
            alloc = numpy.array([a, b, c, d])
            stdev, mean, sharpe_ratio, cum_ret = simulate(dt_start, dt_end, ls_symbols, alloc)
            if sharpe_ratio > max_sharpe_ratio:
                max_sharpe_ratio = sharpe_ratio
                max_data = [dt_start.strftime('%B %d, %Y'), dt_end.strftime('%B %d, %Y'), ls_symbols, alloc, sharpe_ratio, stdev, mean, cum_ret]

print 'Start Date: ' + str(max_data[0])
print 'End Date: ' + str(max_data[1])
print 'Symbols: ' + str(max_data[2])
print 'Optimal Allocations: ' + str(max_data[3])
print 'Sharpe Ratio: ' + str(max_data[4])
print 'Volatility (stdev of daily returns): ' + str(max_data[5])
print 'Average Daily Return: ' + str(max_data[6])
print 'Cumulative Return: ' + str(max_data[7])
