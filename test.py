from models.BootstrapDY import estimate_var, make_dynamics, make_var_mbb, make_boot_dynamics, rolling_window, static, DieboldYilmaz2012
import pandas as pd
import numpy as np
#%%
data = pd.read_csv('data/test/dy2012.csv', parse_dates=['Index'], index_col=0)
YY = data.values

#%%
result = DieboldYilmaz2012(YY, 4, 10)

#%%
from models.funcitons import calcAvgSpilloversTable

#%%1
diffresult = calcAvgSpilloversTable(YY)

#%%
data1 = data.loc['2007-1-1':'2008-1-1']
YY1 = data1.values[:,:3]
onewindowboot = static(YY1, 4, 10, 20, 1000)

#%%
rolling = rolling_window(YY, nlags=4, Horizon=10,windowsize=200, length=20, numboot=100)
#%%
