import rpy2.robjects as robjects

R = robjects.r
R.rm(list=R.ls(all=True))

from py_robustm.run import worker
import pandas as pd
import numpy as np
from py_robustm.logger import LOGGER
import rpy2.robjects.numpy2ri
import json
import pickle
from joblib import Parallel, delayed
from py_robustm.constants import SAVE_DIR
import datetime as dt

robjects.numpy2ri.activate()

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs",
                        default=2 * os.cpu_count() - 1,
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--backend",
                        type=str,
                        default="loky",
                        help="Joblib backend")
    parser.add_argument("--save",
                        action="store_true",
                        help="Save the result")
    args = parser.parse_args()

    if args.save:
        iter = len(os.listdir(SAVE_DIR))
        save_dir = f"{SAVE_DIR}/run_{iter}_{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir)
        LOGGER.info(f"Created save directory at {save_dir}")

    # r.graphics.off()
    # r.source('RobustM_Packages.R')
    # r.source('RobustM_Functions.R')

    ####Parameters for Portfolio Strategies####
    transi = 0.005  # transactional costs ratio
    # end = returns.shape[0]
    lambda_ = 2
    # hel = ret;
    # hel[] = 0

    config = json.load(open('config.json'))
    strats = config["strats"]
    window = config["window"]
    freq = config["freq"]
    verbose = config["verbose"]
    rebalance_dates = config.get("rebalance_dates", None)
    start_test = config.get("start_test", None)
    end_date = config.get("end_date", None)

    # Load data
    prices = pd.read_csv("SP100_20100101_20201231.csv", sep=";")
    prices.set_index('date', 1, inplace=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.astype(np.float32)
    assets = list(prices.columns)
    LOGGER.info(f"\n{prices.head()}")
    LOGGER.info(f"Data shape: {prices.shape}")
    nans = np.array(assets)[(prices.isna().sum() != 0).values.tolist()]
    if len(nans) > 0:
        LOGGER.info(f"Assets with NaNs: {nans}\nFill nans by interpolation")
        # Fillnans
        prices = prices.interpolate(method='polynomial', order=2)
    assert len(np.array(assets)[(prices.isna().sum() != 0).values.tolist()]) == 0
    returns = np.log(prices.pct_change(1).dropna() + 1)
    assert np.sum(returns.isna().sum()) == 0
    dates = list(returns.index)
    prices = prices.loc[dates]

    if start_test is None:
        start_test = dates[window]
        LOGGER.info(f"Start test is not specified by user, starting test at {start_test}")
    else:
        assert len(returns.loc[:start_test].iloc[
                   :-1]) >= window, f"First period is to small for given window {window} and start_test {start_test}"
    if end_date is None:
        end_date = dates[-1]
        LOGGER.info(f"End test is not specified by user, ending test at final date {end_date}")

    test_dates = np.array(dates)[
        [(d >= pd.to_datetime(start_test)) and (d <= pd.to_datetime(end_date)) for d in dates]].tolist()
    if rebalance_dates is None:
        rebalance_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start_test, end=end_date, freq=freq)]

    if args.n_jobs == 1:
        port_weights = []
        for rb_date in rebalance_dates:
            w = worker(returns, rb_date, window, strats, verbose=verbose)
            port_weights.append(w)
    elif args.n_jobs > 1:
        with Parallel(n_jobs=args.n_jobs, backend=args.backend) as _parallel_pool:
            port_weights = _parallel_pool(
                delayed(worker)(returns, rb_date, window, strats, verbose=verbose)
                for rb_date in rebalance_dates
            )
    else:
        raise ValueError(args.n_jobs)

    port_weights = {strat: [r[strat] for r in port_weights] for strat in strats}
    for strat in port_weights:
        port_weights[strat] = pd.DataFrame(port_weights[strat],
                                           columns=assets,
                                           index=pd.to_datetime(rebalance_dates))
        port_weights[strat] = port_weights[strat].reindex(test_dates, axis=0, method='ffill')

    if args.save:
        LOGGER.info(f"Saving results to {save_dir}")
        pickle.dump(open(f"{save_dir}/weights.p", "wb"))

    LOGGER.info("Done")
