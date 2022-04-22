# py-RobustM

Please refer to [RobustM](https://github.com/QuantLet/RobustM) original library. This repository is just a version with
main script in python.

When citing this project, please cite the original paper
from [Härdle et al (2021)](https://www.wiwi.hu-berlin.de/de/forschung/irtg/results/discussion-papers/discussion-papers-2017-1/irtg1792dp2021-018.pdf)
and [Spilak, Härdle; (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4076843).

## Installation

- Refer to `setup.py`. This version has been tested with R==4.0.2 and python==3.7.11. It will probably work with
  python3.8 also.
- First, you need to create a virtual environment, for example with conda: `conda create -n robustm python=3.7`
- Then activate it with `conda activate robustm`
- Install the package in your environment with: `pip install .`

## Run

### Data

Please contact me to get access to the data. Put the csv files in a `data` folder at the root.

### Configuration

First fill the `config.json` file, you can specify:

- `dataset` (str): "dataset1" or "dataset2" to reproduce EmbeddingPortfolio ([Spilak, Härdle; (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4076843))
- `strats` (list): Strategy to evaluate, available strategies
  are `["GMV", "GMV_long", "GMV_lin", "GMV_nlin", "GMV_robust"]`
- `window` (Optional[int]): default is 252 (one year), past window for parameters estimation
- `freq`(Optional[str]): rebalancing freq (should be 'MS', for month). This is not used if `rebalance_dates`is given
- `verbose` (Optional[int]): verbosity, 0 by default
- `rebalance_dates` (Optional[list]): If specificied, then we use those dates to define the rebalancing periods
- `start_test` (Optional[str]): If specificied, this will be the first rebalancing period
- `end_date` (Optional[str]): If specificied, use this for the last testing date
- `name` (Optional[str]): Name of run to save results, `None` by default

### Run

Then just run `main.py` with `python main.py`. You can add command line argument specified in main.py:

- `--save`: to save the results
- `--backend`: specific backend
- `--n_jobs`: number of parallel jobs
