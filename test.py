import rpy2.robjects as robjects
import pandas as pd
import numpy as np
from py_robustm.logger import LOGGER
from py_robustm.run import allocate
import rpy2.robjects.numpy2ri

robjects.numpy2ri.activate()
STRATS = ["GMV", "GMV_long", "GMV_lin", "GMV_nlin", "GMV_robust"]

if __name__ == '__main__':

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

    rett = returns.iloc[-100:, ]
    for strat in STRATS:
        LOGGER.info(strat)
        w = allocate(rett.values, strat=strat)
        LOGGER.info(w)
