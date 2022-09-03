from models.BootstrapDY import estimate_var, make_dynamics, make_var_mbb, make_boot_dynamics, rolling_window, static, DieboldYilmaz2012
import pandas as pd
import numpy as np
import pickle

market_type = 'SectorEquity'
#%%
VaR05 = pd.read_csv('data/'+market_type+'/FullMarkets/VaR05.csv', parse_dates=['date'], index_col=['date'])
VaR05YY = VaR05.values
#%%
VaR05_rolling_150 = rolling_window(VaR05YY, 2, 10, 150, 18)
#VaR05_rolling_250 = rolling_window(VaR05YY, 2, 10, 250, 20)
#%%
Vol = pd.read_csv('data/'+market_type+'/FullMarkets/Vol.csv', parse_dates=['date'], index_col=['date'])
VolYY = Vol.values
#%%
Vol_rolling_150 = rolling_window(VolYY, 1, 10, 150, 18)
#Vol_rolling_250 = rolling_window(VolYY, 1, 10, 250, 20)
#%%
f = open('data\\'+market_type+'\\DYresults\\'+'test.pkl', 'wb')
datas = (VaR05, VaR05_rolling_150, Vol, Vol_rolling_150)
pickle.dump(datas, f, -1)
f.close()

