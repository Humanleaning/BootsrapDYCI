# -*- coding: utf-8 -*-
"""
1) 程序较为计算密集, 使用numba优化
    1.其中使用了大量的for循环, 并不是是因为没有更好的替代函数, 而是numba parallel不支持过于复杂的numpy函数, 同时numba可以优化for循环
    2.if reader want to modify this code, see
      https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#
      to check whether your numpy function is supported by numba
    3.程序使用面向过程编程, 是因为numba对class进行打包的功能比较复杂, 这里如果有机会可以进一步优化
2) 代码中使用了 matrix multiplication operator @, Python版本必须>3.5
3) 程序只使用了最简单的VAR,和GFEVD,
    1.如果要进行其他的变量筛选方法例如ALASSO VAR, 虽然Python里有glmnet包, 但是很难再应用numba速度会慢,推荐使用MATLAB重写程序
        也不是完全不可以用Python, 但是用外生库接入计算VAR1000次都要1分钟一天, 如果计算glmnet估计会更久
    2.如果VAR方程仍然可以用OLS估计,可以通过改动函数实现
    3.如果可以手写函数的估计方法, 可以通过改动函数实现


符号系统
    重要说明:
    这里有个比较致命的冲突:
        1)数据的输入和输出\估计\利用估计值重复计算的时候, 用行观点比较舒服
        2)但是所有数学公式都是用列观点写的
    为了和大多数运算系统保持一致, 本程序的矩阵公式均为行观点模式(即每一行向量为一个日期observation),
    而论文往往为列观点模式(即每一列向量为一个日期observation), 在进行对照时一个较为方便的方法是将论文中所有公式进行左右转置处理
    通常来说, 函数中的变量命名为小写字母更好, 这里采用大写复写的方式, 目的是为了循环中尽量采用一致的循环变量
    TT       表示T, 非滞后项样本数量
    KK       表示K, 变量个数
    pp|nlags 表示p, 滞后阶数
    NN       表示1+K*p, vectorize 后的滞后变量矩阵长度, 包含截距项
    AA       表示系数矩阵, 包含截距项, 为行观点格式, 即此时利用行观点yy = xx*AA
    UU       表示扰动项矩阵, 一般为T×K


This module contains the functions to estimate a VAR model and to compute GFEVD

This module has the following functions:
    estimate_var                      (估计参数)
    make_dynamics                     (根据估计的参数计算GFEVD)
    make_boot_dynamics                (根据抽取的扰动项计算出一个样本, 调用函数估计参数计算GFEVD)
    make_var_mbb                      (实现由一个模块进行扰动项的抽取)
    rolling_window                    (实现rolling window, 即将数据分发成不同的window, 带入make_var_mbb)
    static                            (实现静态的bootstrap)


The main functions are
    make_proxy_svar_mbb : Uses MBB to produce confidence intervals
The other functions exist to support these funtions.

This module uses the numpy and numba packages.
Several functions in this module use the @njit decorator from numba.
"""

# Load the packages
import numpy as np
from numba import njit, prange
#%%
@njit
def estimate_var(yy, xx):
    """
    Uses endog and exog data to estimate the parameters of a VAR.

    :param yy: endog variable, T × K array
    :param xx: exog variable, T × 1+K*p array

    :return: AA_est        列观点含截距项系数
    :return: U_est      行观点误差矩阵
    :return: covUU_est  相关系数矩阵
    """

    # Dimensions of the data
    TT, KK = yy.shape                                    #
    NN = xx[0, :].size                                   # 这里的NN为1+p*K, 方程本质上是N个regressor, K个regressand

    ############## Estimate VAR coefficients by least squares#######################
    #给出了3种方法
    ####solution1####:numpy.linalg.solve()[这个版本是论文开源库版本, 问题在于@运算有转置符号, 属于非连续储存矩阵, 运算速度较慢]
    ####see: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    # mat1 = np.zeros((NN, NN))                            # 这里即构成N个regressor的矩阵, 注意因为linalg.solve需要square matrix这里平方
    # mat1[:, :] = xx.T @ xx                               # xx' * xx
    # mat2 = np.zeros((NN, KK))                            # 这集即构成K个regressand的矩阵, 注意这里同样增加了xx',使得解的结果正确
    # mat2[:, :] = xx.T @ yy                               # xx' * yy
    # A_est = np.zeros((NN, KK))                           # A_est的维度即N * K, 从而 T×K(yy) = T×N(xx) * N×K(A_est)
    # A_est = np.linalg.solve(mat1, mat2)                  # Solves the linear equation set mat1 * x = mat2 for the unknown x for square a matrix.
    # # VAR residuals
    # U_est = np.zeros((TT, KK))                           # 新的残差项
    # U_est[:, :] = yy - xx @ A_est                        # 记住, 这个A_est的使用方式是, xx @ A_est
    ####solution2####:numpy.linalg.lstsq()[这个版本是按照statsmodel来进行的]
    ####see: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    ####这个感觉有冗余计算
    A_est = np.linalg.lstsq(xx, yy, rcond=1e-15)[0]      #TT×KK = TT×NN * NN*KK
    # VAR residuals
    U_est = np.zeros((TT, KK))
    U_est[:, :] = yy - xx @ A_est
    ####solution3####:Lütkepohl pp. 146-153
    ####讲道理这个应该最快, 符号系统参考Inference in VARs with Conditional Heteroskedasticity of Unknown Form
    ####但是要回去查文献, 如果速度不能接受在想这个办法


    # Covariance matrices                                # 这里回顾一下OLS的一阶条件是扰动项均值为0
    covUU_est = np.zeros((KK, KK))                       # K个变量之间的相关系数,因为均值为0, 所以只需要向量相乘即可, 注意要除以T
    covUU_est[:, :] = (U_est.T @ U_est) / (TT-NN)        # 这里是按照修正版本计算

    # Transpose A
    # A_est = A_est.T                                    # 输出行观点系数,  从N×K变为K×N, 即K×(1+p*K) ###不进行这步, 则输出列观点系数

    return A_est, U_est, covUU_est

@njit
def make_dynamics(AA, CovUU, pp, Horizon):
    """
    Uses the estimated parameters of a VAR to produce GIRFs and GFEVDs.

    :param AA:      行观点含截距项系数矩阵
    :param CovUU:   相关系数矩阵
    :param pp:      滞后阶数
    :param Horizon: 预测H

    :return:        normalized Generalized Decomposition Matrix
    """

    # Number of VAR variables
    NN, KK = AA.shape                                       # AA这里是行观点系数
    #coefs = AA[1:, :].reshape((pp, KK, KK))                # 1.提取非截距项部分 2.为了计算方便,将系数矩阵设为3维

    coefs = np.zeros((pp, KK, KK))
    for i in range(pp):
        coefs[i, :, :] = AA[1+i*KK:1+(i+1)*KK, :]

    Phis = np.zeros((Horizon, KK, KK))
    Phis[0] = np.eye(KK)

    # recursively compute Phi matrices
    for i in range(1, Horizon):
        for j in range(1, i+1):
            if j > pp:
                break

            Phis[i] += coefs[j-1] @ Phis[i-j]  # 注意这里Phis的下标没有位移(从Phi0开始), 但是AA的下标有位移


    # Compute H-step generalized variance decomposition
    deviations = np.sqrt(np.diag(CovUU))
    numerators = np.zeros((Horizon, KK, KK))
    denominator = np.zeros((Horizon, KK, KK))

    ###############################################################################################
    for h in range(Horizon):
        #这里进行公式，定义分子和分母
        numerators[h] = Phis[h].T @ CovUU
        denominator[h] = numerators[h] @ Phis[h]
        numerators[h] = np.square(numerators[h] / deviations)

    DD = np.zeros((KK, KK))
    for i in range(KK):
        for j in range(KK):
            DD[i, j] = np.sum(numerators[:, i, j]) / np.sum(denominator[:, i, i])
    ##################################################################################################

    for h in range(Horizon):
        denominator[h] = Phis[h].T @ CovUU @ Phis[h]
        for i in range(KK):
            I_matrix = np.eye(KK)
            Mi = np.hstack((I_matrix[:,:i],I_matrix[:,i+1:]))
            numeratorsi = Phis[h].T @ CovUU @ Mi @ np.linalg.inv(Mi.T @ CovUU @ Mi) @ Mi.T @ CovUU @ Phis[h]
            numerators[h,:,i] = np.diag(numeratorsi)
    dd = np.zeros((KK, KK))
    for i in range(KK):
        for j in range(KK):
            dd[i,j] = np.sum(numerators[:,i,j]) / np.sum(denominator[:,i,i])
    dd = 1-dd

    Gap = DD-dd
    #normalize
    #DD_tilde = DD/DD.sum(axis=1).reshape(KK, -1)*100
    # DD_tilde = DD
    return Gap

@njit
def make_boot_dynamics(AA, u_in, y_init, nlags, Horizon):
    """
    Uses resampled VAR residuals to compute bootstrapped IRFs and FEVDs.

    This function calls:
        estimate_var
        make_dynamics

    :param AA:     行观系数矩阵
    :param u_in:     列观点扰动项
    :param y_init:   列观点展平成x类型的y
    :param nlags:    滞后阶数
    :param Horizon:  预测周期

    :return:dd_tilde_star     array K×K
    """

    # Dimensions of the data
    TT, KK = u_in.shape                                              # 这里输入第i次抽样得到的UU

    # Construct the VAR variables (loop from 1-T-1 day)
    x_star = np.zeros((TT, 1+KK*nlags))                          # x_star是对应的外生变量, 即滞后项, 以展平的方式输入
    x_star[0, :] = y_init[:]                                         # 第一天是直接被给予的
    y_star = u_in                                                    # 直接设置为u_in, 好处是每次是在其上增加x*A
    for ii in range(TT - 1):                                         # 这里先计算前T-1天, 注意最后一天因为不需要更新x_star而分开计算
        # Compute hist @ AA
        for kk in range(KK):                                         # 对小型运算用numba优化for要比numpy矩阵*向量更优
            for nn in range(KK * nlags + 1):
                y_star[ii, kk] = y_star[ii, kk] + x_star[ii, nn] * AA[nn, kk]

        # Update X_star
        x_star[ii + 1, KK + 1:KK * nlags + 1] = x_star[ii, 1:KK * (nlags - 1) + 1]
        x_star[ii + 1, 1:KK + 1] = y_star[ii, :]                     # 从这里可以看出, x_star是1排在最左边, 新的滞后项依次右列
        x_star[ii + 1, 0] = 1

    # Construct the VAR variables (Last observation day)
    for kk in range(KK):                                             # 这里计算最后一天, 注意只计算了y_star而不需再更新x_star
        for nn in range(KK * nlags + 1):
            y_star[TT - 1, kk] = y_star[TT - 1, kk] + x_star[TT - 1, nn] * AA[nn, kk]

    # Estimate the proxy SVAR
    a_star, u_starhat, covUU_star = estimate_var(y_star, x_star)


    # Compute the IRFs, FEVDs, and statistics for AR-type confidence sets
    dd_tilde_star = make_dynamics(a_star, covUU_star, nlags, Horizon)


    return dd_tilde_star

@njit
def make_var_mbb(AA, UU, Yinit, nlags, Horizon, numboot, length):
    """
    Runs the moving block bootstrap and compute fevd matrix

    This function calls:
        make_boot_dynamics

    :param AA:                   # 行观点系数矩阵 array  K × (1+K*p)
    :param UU:                   # 行观点误差矩阵 array  T × K
    :param Yinit:                # 行观点滞后变量, 其实就是xx的第一行非常数项部分
    :param nlags:                # 滞后阶数
    :param Horizon:              # forecast horizon
    :param numboot:              # bootstrap的次数
    :param length:               # 即l, 一个block的长度

    :return: fevd_store          # normalized之后的DD, array K × K × numboot
    """

    # Dimensions of the data
    TT, KK = UU.shape                                # TT表示日期数, 即windowsize-nlags

    # Construct the blocks
    numblocks = TT - length + 1                      # numblocks: 一共需要有numblocks个抽样总体
    u_blocks = np.zeros((length, KK, numblocks))     # 预设抽样总体, 其中每一个矩阵为l*K维度
    for ii in range(numblocks):                      # 装填抽样总体
        u_blocks[:, :, ii] = UU[ii:length + ii, :]

    # Centering for the blocks                       # u_tilde^star到u^star时, 矩阵需要中心化
    u_temp = np.zeros((length, KK))                  # 预设中心化时的减项的重复因子, 共有l行不一样的元素
    for ii in range(length):                         # 装填中心化时的减项的重复因子
        for nn in range(KK):
            u_temp[ii, nn] = np.mean(UU[ii:TT - length + ii + 1, nn])

    numResample = int(np.ceil(TT / length))          # 共抽取numResample个样本
    u_center = np.zeros((numResample * length, KK))  # 预设中心化时的减项矩阵, 即把u_temp重复numResample次
                                                     # 这一步可以改为用(numpy.kron() (‘C’ and ‘F’ order only))
    for ii in range(numResample):                    # 装填中心化时的减项矩阵
        u_center[ii * length:(ii + 1) * length, :] = u_temp

    # Set dimensions of arrays to hold bootstrapped estimates
    fevd_store = np.zeros((KK, KK, numboot))


    # Bootstrap loop
    for boot in range(numboot):
        # Resample the blocks
        ratio = int(np.ceil(TT / length))
        u_temp = np.empty((length * ratio, KK))
        for ii in range(ratio):
            index = np.random.randint(numblocks)
            u_temp[ii * length:(ii + 1) * length, :] = u_blocks[:, :, index]


        # Center the re-sampled variables
        u_temp[:, :] = u_temp - u_center

        # Truncate to get bootstrapped U and M
        u_star = np.empty((TT, KK))
        u_star[:, :] = u_temp[0:TT, :]

        # Compute bootstrapped FEVDs
        fevd_store[:, :, boot] = make_boot_dynamics(AA, u_star, Yinit, nlags, Horizon)

    return fevd_store

#%%
@njit(parallel=True)
def rolling_window(YY, nlags:int, Horizon:int, windowsize:int, length:int, numboot:int=1000):

    # 先把整体的yy给slice一下
    numObs, KK = YY.shape
    fTT = numObs - nlags

    fyy = YY[nlags:, :]
    fxx = np.zeros((fTT, 1 + nlags*KK))
    fxx[:, 0] = 1
    for t in range(fTT):
        for p in range(nlags):
            fxx[t, 1+p*KK:1+KK+p*KK] = YY[t+nlags-p-1, :]

    # 先把数据拆分成非常多个window
    numWindows = numObs - windowsize + 1

    # GEV
    GFEVD = np.zeros((numWindows, KK, KK, numboot))
    #下面开始并行运算:
    for i in prange(numWindows):
        yy = fyy[i:i+windowsize-nlags]
        xx = fxx[i:i+windowsize-nlags]
        # 对每天的数据, 先估计一遍var
        A_est, U_est, covUU_est = estimate_var(yy, xx)
        # 然后带入make_var_mbb, 进行GFEVD估计, 然后把结果传入一个矩阵
        GFEVD[i, :, :, :] = make_var_mbb(A_est, U_est, xx[0, :], nlags, Horizon, numboot, length)

    return GFEVD

def static(YY, nlags:int, Horizon:int, length:int=None, numboot:int=1000):
    # 先把整体的yy给slice一下
    numObs, KK = YY.shape
    fTT = numObs - nlags
    length = np.round(5.03*np.power(fTT, 0.25)) if length is None else length

    fyy = YY[nlags:, :]
    fxx = np.zeros((fTT, 1 + nlags*KK))
    fxx[:, 0] = 1
    for t in range(fTT):
        for p in range(nlags):
            fxx[t, 1+p*KK:1+KK+p*KK] = YY[t+nlags-p-1, :]


    A_est, U_est, covUU_est = estimate_var(fyy, fxx)
    GFEVD = make_var_mbb(A_est, U_est, fxx[0, :], nlags, Horizon, numboot, length)

    return GFEVD

def DieboldYilmaz2012(YY, nlags:int, Horizon:int):
    # 先把整体的yy给slice一下
    numObs, KK = YY.shape
    fTT = numObs - nlags

    fyy = YY[nlags:, :]
    fxx = np.zeros((fTT, 1 + nlags*KK))
    fxx[:, 0] = 1
    for t in range(fTT):
        for p in range(nlags):
            fxx[t, 1+p*KK:1+KK+p*KK] = YY[t+nlags-p-1, :]


    A_est, U_est, covUU_est = estimate_var(fyy, fxx)
    GFEVD = make_dynamics(A_est, covUU_est, nlags, Horizon)

    return GFEVD











