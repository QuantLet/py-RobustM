import rpy2.robjects as robjects
import pandas as pd
import numpy as np
from py_robustm.logger import LOGGER
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
from py_robustm import markowitz as mrkw

robjects.numpy2ri.activate()



if __name__ == '__main__':
    print(rpy2.__version__)

    # import R's "base" package
    # base = importr('base')
    # utils = importr('utils')

    r = robjects.r
    # r.graphics.off()
    r.rm(list=r.ls(all=True))
    # r.source('RobustM_Packages.R')
    # r.source('RobustM_Functions.R')
    prices = pd.read_csv("SP100_20100101_20201231.csv", sep=";")
    LOGGER.info(prices.head())
    prices.set_index('date', 1, inplace=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.astype(np.float32)
    assets = list(prices.columns)
    LOGGER.info(f"Data shape: {prices.shape}")
    nans = np.array(assets)[(prices.isna().sum() != 0).values.tolist()]
    LOGGER.info(f"Assets with NaNs: {nans}\nFill nans by interpolation")
    # Fillnans
    prices = prices.interpolate(method='polynomial', order=2)
    assert len(np.array(assets)[(prices.isna().sum() != 0).values.tolist()]) == 0
    returns = np.log(prices.pct_change(1).dropna() + 1)
    assert np.sum(returns.isna().sum()) == 0
    dates = returns.index
    prices = prices.loc[dates]

    ####Parameters for Portfolio Strategies####
    freq = "months"
    transi = 0.005  # transactional costs ratio
    end = returns.shape[0]
    lambda_ = 2
    # hel = ret;
    # hel[] = 0

    window = 252  # one year
    start_date = '2020-01-01'
