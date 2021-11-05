import rpy2.robjects as robjects

R = robjects.r
R.rm(list=R.ls(all=True))

from py_robustm.py_robustm import worker, load_data, evaluate
import pandas as pd
import numpy as np
from py_robustm.logger import LOGGER
import rpy2.robjects.numpy2ri
import json
import pickle
from joblib import Parallel, delayed
from py_robustm.constants import SAVE_DIR
import datetime as dt
import time

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
    LOGGER.info(f"Starting script with {args.n_jobs} jobs in parallel...")
    config = json.load(open('config.json'))
    LOGGER.info(f"Config loaded")
    if args.save:
        if os.path.exists(SAVE_DIR):
            iter_ = len(os.listdir(SAVE_DIR))
        else:
            os.mkdir(SAVE_DIR)
            iter_ = 0
        name = config.get("name")
        if name is None:
            name = config["dataset"]
        save_dir = f"{SAVE_DIR}/run_{iter_}_{name}_{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        os.mkdir(save_dir)
        LOGGER.info(f"Created save directory at {save_dir}")
        json.dump(config, open(f"{save_dir}/config.json", "w"))
        LOGGER.debug(f"Config saved")

    ####Parameters for Portfolio Strategies####
    transi = 0.005  # transactional costs ratio
    # end = returns.shape[0]
    lambda_ = 2
    # hel = ret;
    # hel[] = 0

    dataset = config["dataset"]
    strats = config["strats"]
    window = config["window"]
    freq = config["freq"]
    verbose = config.get("verbose", 0)
    rebalance_dates = config.get("rebalance_dates", None)
    if rebalance_dates is not None:
        start_test = rebalance_dates[0]
    else:
        start_test = config.get("start_test", None)
    end_date = config.get("end_date", None)
    random_stocks = config.get("random_stocks", False)

    # Load data
    LOGGER.info(f"Loading dataset: {dataset}")
    prices, returns = load_data(dataset, random_stocks=random_stocks)
    assets = list(prices.columns)
    dates = list(returns.index)

    if end_date is None:
        end_date = dates[-1]
        LOGGER.info(f"End test is not specified by user, ending test at final date {end_date}")

    if start_test is None:
        start_test = dates[window]
        LOGGER.info(f"Start test is not specified by user, starting test at {start_test}")
    else:
        assert pd.to_datetime(start_test) < end_date, f"start_test: {start_test} and end_date is: {end_date}"
        assert len(returns.loc[:start_test].iloc[
                   :-1]) >= window, f"First period is to small for given window {window} and start_test {start_test}"

    test_dates = np.array(dates)[
        [(d >= pd.to_datetime(start_test)) and (d <= pd.to_datetime(end_date)) for d in dates]].tolist()
    if rebalance_dates is None:
        rebalance_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start_test, end=end_date, freq=freq)]
    LOGGER.info(f"There are {len(rebalance_dates)} rebalancing periods")
    LOGGER.info(f"Computing weights...")
    t1 = time.time()
    if args.n_jobs == 1:
        port_weights = []
        for rb_date in rebalance_dates:
            w = worker(returns, rb_date, window, strats, verbose=verbose)
            port_weights.append(w)
    elif args.n_jobs > 1:
        with Parallel(n_jobs=args.n_jobs, backend=args.backend) as _parallel_pool:
            port_weights = _parallel_pool(
                delayed(worker)(returns, rb_date, window, strats, verbose=verbose, to_go=len(rebalance_dates) - i - 1)
                for i, rb_date in enumerate(rebalance_dates)
            )
    else:
        raise ValueError(args.n_jobs)
    t2 = time.time()
    LOGGER.info(f"Time to compute weights: {round((t2 - t1) / 60, 2)} min.")

    port_weights = {strat: [r[strat] for r in port_weights] for strat in strats}
    for strat in port_weights:
        port_weights[strat] = pd.DataFrame(port_weights[strat],
                                           columns=assets,
                                           index=pd.to_datetime(rebalance_dates))
        port_weights[strat] = port_weights[strat].reindex(test_dates, axis=0, method='ffill')

    if args.save:
        LOGGER.info(f"Saving results to {save_dir}")
        pickle.dump(port_weights, open(f"{save_dir}/weights.p", "wb"))

    LOGGER.info("Evaluate performance")
    if args.save:
        evaluate(returns, port_weights, save_dir=save_dir)
    else:
        evaluate(returns, port_weights)

    LOGGER.info("Done")
