from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numpy as np
import pandas as pd
from py_robustm.logger import LOGGER
from typing import Dict, List
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
import py_robustm.markowitz as mrkw
from py_robustm.constants import AVAILABLE_STRATS

robjects.numpy2ri.activate()  # For numpy to R object conversion
R = robjects.r
rp = importr('RiskPortfolios')
nlshrink = importr('nlshrink')


def load_data(dataset: str):
    """
    Load data
    :param dataset: name of dataset
    :return:
    """
    if dataset == "SP100":
        prices = pd.read_csv("data/SP100_20100101_20201231.csv", sep=";")
        prices.set_index('date', inplace=True)
    elif dataset == "Russell3000":
        prices = pd.read_csv("data/Russell3000prices_20000101_20201231.csv")
        prices.set_index('date', inplace=True)
    else:
        raise NotImplementedError(dataset)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.astype(np.float32)
    LOGGER.info(f"\n{prices.head()}")
    LOGGER.info(f"Data shape: {prices.shape}")
    assets = list(prices.columns)
    nans = np.array(assets)[(prices.isna().sum() != 0).values.tolist()]
    if len(nans) > 0:  # Fill nans
        LOGGER.info(f"Assets with NaNs: {nans}")
        LOGGER.info("Fill nans by interpolation")
        prices = prices.interpolate(method='polynomial', order=2)
    assert len(np.array(assets)[(prices.isna().sum() != 0).values.tolist()]) == 0
    returns = np.log(prices.pct_change(1).dropna() + 1)
    assert np.sum(returns.isna().sum()) == 0
    prices = prices.loc[returns.index]

    return prices, returns


def allocate(returns: np.ndarray, strat: str, **kwargs: Dict):
    """
    General caller for portfolio allocation for all strats defined in AVAILABLE_STRATS
    :param returns:
    :param strat:
    :param kwargs:
    :return:
    """
    assert strat in AVAILABLE_STRATS, f"Strat: '{strat}' is not implemented. Available strats are {AVAILABLE_STRATS}"
    if strat == "GMV":
        Sigma = rp.covEstimation(returns)
        control = rlc.TaggedList(["minvol", "none"], tags=('type', 'constraint'))  # Named list in R
        w1 = rp.optimalPortfolio(Sigma=Sigma, control=control)

    elif strat == "GMV_long":
        Sigma = rp.covEstimation(returns)
        control = rlc.TaggedList(["minvol", "lo"], tags=('type', 'constraint'))  # Named list in R
        w1 = rp.optimalPortfolio(Sigma=Sigma, control=control)

    elif strat == "GMV_lin":
        Sigma_lin = nlshrink.linshrink_cov(returns, k=0)
        control = rlc.TaggedList(["minvol"], tags=('type',))
        w1 = rp.optimalPortfolio(Sigma=Sigma_lin, control=control)

    elif strat == "GMV_nlin":
        Sigma_nlin = nlshrink.nlshrink_cov(returns, k=0)
        control = rlc.TaggedList(["minvol"], tags=('type',))
        w1 = rp.optimalPortfolio(Sigma=Sigma_nlin, control=control)

    elif strat == "GMV_robust":
        Sigma = rp.covEstimation(returns)
        gammas = mrkw.recommended_gamma(Sigma)
        block_n = kwargs.get("block_n", 5)
        w1 = mrkw.robust_qp(returns, block_n, gamma=gammas, lmbd=0)
    else:
        raise NotImplementedError(f"Strat: '{strat}' is not implemented")

    return w1


def worker(returns: pd.DataFrame, date: str, window: int, strats: List, verbose: int = 0, **kwargs) -> Dict:
    """

    :param returns: dataframe of log returns
    :param date: rebalancing date
    :param window: length of past window used for estimation
    :param strats: list of portfolio allocation strategies
    :param verbose: print logs are not
    :return: Dictionary of strategy weights
    """
    to_go = kwargs.get('to_go')
    if to_go is not None:
        LOGGER.info(f"Steps to go: {to_go}")
    if verbose > 0:
        LOGGER.info(f"Rebalancing at date {date}")
    past_returns = returns.loc[:date].iloc[:-1]  # remove last row which includes rb_date
    past_returns = past_returns.iloc[-window:]  # take only sample of size window
    if verbose > 0:
        LOGGER.info(f"Using past returns from {past_returns.index[0]} to {past_returns.index[-1]}")
    weights = {}
    for strat in strats:
        if verbose > 0:
            LOGGER.info(f"Computing weights for allocation method: {strat} ")
        w = allocate(past_returns.values, strat=strat)
        weights[strat] = w

    return weights
