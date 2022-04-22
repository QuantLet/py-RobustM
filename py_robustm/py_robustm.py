from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numpy as np
import pandas as pd
from py_robustm.logger import LOGGER
from typing import Dict, List, Optional
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
import py_robustm.markowitz as mrkw
from py_robustm.constants import AVAILABLE_STRATS, AVAILABLE_DATASET
import matplotlib.pyplot as plt

robjects.numpy2ri.activate()  # For numpy to R object conversion
R = robjects.r
rp = importr('RiskPortfolios')
nlshrink = importr('nlshrink')


def load_dataset1():
    data = pd.read_csv("data/dataset1.csv", index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.astype(np.float32)
    return data, list(data.columns)


def load_dataset2():
    data = pd.read_csv('data/dataset2.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.interpolate(method='polynomial', order=2)
    data = data.astype(np.float32)
    assets = list(data.columns)

    return data, assets


def load_data(dataset):
    assert dataset in AVAILABLE_DATASET, f"Dataset: '{dataset}' is not implemented. Available dataset are {AVAILABLE_DATASET}"
    if dataset == 'dataset1':
        data, assets = load_dataset1()
    elif dataset == 'dataset2':
        data, assets = load_dataset2()
    else:
        raise NotImplementedError(f"dataset must be one of ['dataset1', 'dataset2']: {dataset}")

    return data, assets


def load_data(dataset: str):
    """
    Load data
    :param dataset: name of dataset
    :param random_stocks: Select 600 random stocks for Russell3000 dataset
    :return:
    """
    assert dataset in AVAILABLE_DATASET, f"Dataset: '{dataset}' is not implemented. Available dataset are {AVAILABLE_DATASET}"
    if dataset == "dataset1":
        prices, _ = load_dataset1()
    elif dataset == "dataset2":
        prices, _ = load_dataset2()
    else:
        raise NotImplementedError(dataset)

    assert np.sum(prices.isna().sum()) == 0
    returns = np.log(prices.pct_change(1) + 1).iloc[1:]  # drop first row with nan
    assert np.sum(returns.isna().sum()) == 0
    prices = prices.loc[returns.index]

    LOGGER.info(f"\n{prices.head()}")
    LOGGER.info(f"Data shape: {prices.shape}")
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

    elif strat == "MeanVar_long":
        Sigma = rp.covEstimation(returns)
        mu = rp.meanEstimation(returns)
        control = rlc.TaggedList(["mv", "lo"], tags=('type', 'constraint'))  # Named list in R
        w1 = rp.optimalPortfolio(Sigma=Sigma, mu=mu, control=control)

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


def evaluate(returns: pd.DataFrame, weights: Dict, save_dir: Optional[str] = None, show: bool = False):
    """
    Evaluation of strategies:
        - compute portfolio return and equity
        - graph of equity
    :param returns:
    :param weights:
    :return:
    """
    strats = list(weights.keys())
    start = weights[strats[0]].dropna().index[0]
    port_ret = pd.DataFrame()
    for strat in weights:
        port_ret[strat] = (returns * weights[strat]).sum(1)
    port_ret['EW'] = returns.mean(1)
    port_ret = port_ret.loc[start:]
    port_equity = 1 + port_ret.cumsum()

    plt.figure(figsize=(20, 10))
    for c in port_equity.columns:
        plt.plot(port_equity[c], label=c)
    plt.legend()
    if save_dir:
        plt.savefig(f"{save_dir}/cum_returns.png", bbox_inches="tight")
    if show:
        plt.show()
