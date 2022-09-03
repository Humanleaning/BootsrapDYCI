from statsmodels.tsa.api import VAR
import numpy as np
def calcAvgSpilloversTable(data, forecast_horizon=10, lag_order=4):
    """
    估计GFEVD矩阵
    :param data: 数据, T*K格式
    :param forecast_horizon: int
    :param lag_order: int
    :return:
    """
    # ===
    # sources:
    # https://www.statsmodels.org/dev/vector_ar.html
    # https://en.wikipedia.org/wiki/n#Comparison_with_BIC
    # https://groups.google.com/g/pystatsmodels/c/BqMqOIghN78/m/21NkPAEPJgIJ
    # ===
    model = VAR(data)
    results = model.fit(lag_order)

    sigma_u = np.asarray(results.sigma_u)
    sd_u = np.sqrt(np.diag(sigma_u))

    fevd = results.fevd(forecast_horizon, sigma_u / sd_u)
    fe = fevd.decomp[:, -1, :]
    fevd = (fe / fe.sum(1)[:, None] * 100)

    # cont_incl = fevd.sum(0)
    # cont_to = fevd.sum(0) - diag(fevd)
    # cont_from = fevd.sum(1) - diag(fevd)
    # spillover_index = 100 * cont_to.sum() / cont_incl.sum()
    #
    # names = model.endog_names
    # spilloversTable = pd.DataFrame(fevd, columns=names).set_index([names])
    # spilloversTable.loc['Cont_To'] = cont_to
    # spilloversTable.loc['Cont_Incl'] = cont_incl
    # spilloversTable = pd.concat([spilloversTable, pd.DataFrame(cont_from, columns=['Cont_From']).set_index([names])],
    #                             axis=1)
    # spilloversTable = pd.concat(
    #     [spilloversTable, pd.DataFrame(cont_to - cont_from, columns=['Cont_Net']).set_index([names])], axis=1)
    # spilloversTable.loc['Cont_To', 'Cont_From'] = cont_to.sum()
    # spilloversTable.loc['Cont_Incl', 'Cont_From'] = cont_incl.sum()
    # spilloversTable.loc['Cont_Incl', 'Cont_Net'] = spillover_index
    return fevd

#%%
# import pandas as pd
# import random
# data = pd.read_csv(r'F:\TailRiskConnectedness\results\SectorVaR05.csv', parse_dates=['date'], index_col=['date'])
# data1 = pd.read_csv(r'F:\TailRiskConnectedness\results\SectorVaR05.csv', usecols=[i for i in range(1,12)])
# a = list(data1.columns)
# random.shuffle(a)
# data2 = data1[a]
# # #%%
# fevd, spill = calcAvgSpilloversTable(data)
# fevd1, spill1 = calcAvgSpilloversTable(data1) # test index is irrelevant
# fevd2, spill2 = calcAvgSpilloversTable(data2) # test variable order is irrelevant

