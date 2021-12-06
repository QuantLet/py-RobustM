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


def load_global_bond_data(crypto_assets=['BTC', 'DASH', 'ETH', 'LTC', 'XRP']):
    data = pd.read_csv('./data/alla_data_20211101.csv')
    data = data.interpolate(method='polynomial', order=2)
    data = data.set_index('Date')
    data.index = pd.to_datetime(data.index)
    data = data.dropna()
    crypto_data = pd.read_csv('./data/clean_data_D_20150808_20211102.csv')
    crypto_data = crypto_data.set_index('date')
    crypto_data.index = pd.to_datetime(crypto_data.index)
    crypto_data = crypto_data[crypto_assets]

    data = pd.concat([data, crypto_data], 1).dropna()
    data = data.astype(np.float32)
    return data


def load_multiasset_traditional():
    data = pd.read_csv('data/bloomberg_comb_update_2021.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.interpolate(method='polynomial', order=2)
    data = data.astype(np.float32)

    return data


def load_data(dataset: str, random_stocks: Optional[bool] = False):
    """
    Load data
    :param dataset: name of dataset
    :param random_stocks: Select 600 random stocks for Russell3000 dataset
    :return:
    """
    assert dataset in AVAILABLE_DATASET, f"Dataset: '{dataset}' is not implemented. Available dataset are {AVAILABLE_DATASET}"
    if dataset == "SP100":
        prices = pd.read_csv("data/SP100_20100101_20201231.csv")
        prices.set_index('date', inplace=True)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.astype(np.float32)
    elif dataset == "Russell3000":
        prices = pd.read_csv("data/Russell3000prices_20000101_20201231.csv")
        prices.set_index('date', inplace=True)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.astype(np.float32)
    elif dataset == "global_bond":
        prices = load_global_bond_data()
    elif dataset == "multiasset_traditional":
        prices = load_multiasset_traditional()
    else:
        raise NotImplementedError(dataset)
    assets = list(prices.columns)
    if dataset == 'Russell3000':
        prices = prices.loc["2010-01-01":, :]  # From RobustM repo
    if dataset == 'SP100':
        # There is not much NaNs in this dataset, we can just interpolate
        nans = np.array(assets)[(prices.isna().sum() != 0).values.tolist()]
        if len(nans) > 0:  # Fill nans
            LOGGER.info(f"Assets with NaNs: {nans}")
            print(prices[nans].isna().sum() / len(prices) * 100)
            LOGGER.info("Fill nans by interpolation")
            prices = prices.interpolate(method='polynomial', order=2)
    elif dataset == 'Russell3000':
        raise NotImplementedError()
        prices = prices.dropna()
        if random_stocks:
            LOGGER.info(f"Sampling 600 random stocks from Russell3000")
            random_stocks = np.random.choice(list(prices.columns), size=600, replace=False)
            assert len(np.unique(random_stocks)) == 600
            prices = prices[random_stocks]
    elif dataset == 'global_bond':
        pass
    elif dataset == 'multiasset_traditional':
        pass
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
