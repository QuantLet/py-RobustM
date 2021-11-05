from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numpy as np
from typing import Dict
import rpy2.robjects.numpy2ri
import rpy2.rlike.container as rlc
import py_robustm.markowitz as mrkw
from config import AVAILABLE_STRATS

robjects.numpy2ri.activate()  # For numpy to R object conversion
R = robjects.r
rp = importr('RiskPortfolios')
nlshrink = importr('nlshrink')


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
