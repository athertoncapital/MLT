import pandas as pd
import numpy as np
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import sys
    
cash = int(sys.argv[1])
orders = sys.argv[2]
values = sys.argv[3]
orders = pd.read_csv(orders, parse_dates = [[0, 1, 2]],
                     header = None, index_col = [0])
orders.columns = ['Symbol', 'Command', 'Value', 'Dummy']
orders = orders.sort_index()
s = orders.ix[0].name.strftime('%m-%d-%Y %H:%M:%S')
dt_start = dt.datetime.strptime(s, '%m-%d-%Y %H:%M:%S')
s = orders.ix[len(orders) -1].name.strftime('%m-%d-%Y %H:%M:%S')
dt_end = dt.datetime.strptime(s, '%m-%d-%Y %H:%M:%S') + dt.timedelta(days = 1)

c_dataobj = da.DataAccess('Yahoo')
ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
dt_timeofday = dt.timedelta(hours=16)
ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
symbols = list(set(orders.Symbol.tolist()))
ldf_data = c_dataobj.get_data(ldt_timestamps, symbols, ls_keys)
d_data = dict(zip(ls_keys, ldf_data))

stock_prices = d_data['close'].values

dates = []
for i in range(len(orders)):   
    dates.append(orders.ix[i].name)
    
output = ""
stock_cnt = {}
for i in range(len(ldt_timestamps)):
    date = ldt_timestamps[i] - dt.timedelta(hours = 16)
    if date in dates:
        matches = [j for j in range(len(dates)) if date == dates[j]]
        for date_i in matches:
            sym = orders.ix[date_i,0]
            sym_i = symbols.index(sym)
            cmd = orders.ix[date_i,1]
            val = orders.ix[date_i,2]
            if sym not in stock_cnt:
                if cmd == "Buy":
                    stock_cnt[sym] = val
                    cash = cash - (stock_prices[i][sym_i]*val)
                else:
                    stock_cnt[sym] = -val
                    cash = cash + (stock_prices[i][sym_i]*val)
            else:
                if cmd == "Buy":
                    stock_cnt[sym] = stock_cnt.get(sym) + val
                    cash = cash - (stock_prices[i][sym_i]*val)
                else:
                    stock_cnt[sym] = stock_cnt.get(sym) - val
                    cash = cash + (stock_prices[i][sym_i]*val)
    asset_val = 0
    for sym in symbols:
       if sym in stock_cnt:
            asset_val += stock_cnt[sym]*stock_prices[i][symbols.index(sym)]
    
    output += str(date.strftime('%Y, %m, %d, ')) + str(cash + asset_val) + "\n"
f = open(values, 'w')
f.write(output)
