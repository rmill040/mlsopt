import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Package imports 
from mlsopt.samplers import (BernoulliFeatureSampler, PipelineSampler, 
                             XGBClassifierSampler)
from mlsopt.optimizers import RSOptimizer
from mlsopt.utils import STATUS_OK

# Constants
SEED = 1718
MAIN_DIR    = Path(__file__).resolve().parents[1]
# RESULTS_DIR = 
import pdb; pdb.set_trace()

def rs_vs_rsdu():
    """Create plot of random search vs random search with updates
    """
    # Load toy data
    X, y = load_breast_cancer(return_X_y=True)

    # Define samplers
    hp_sampler = XGBClassifierSampler(dynamic_update=True, seed=SEED)
    sampler    = PipelineSampler(seed=SEED)\
                    .register_sampler(hp_sampler, name='hp')

    def fn(space):
        """Function to optimize.
        """
        clf    = XGBClassifier(**space['hp'])
        metric = cross_val_score(clf, X, y, cv=5, n_jobs=5).mean()
        return {
            'status'  : STATUS_OK,
            'metric'  : metric,
            'message' : ''
        }

    # Random search without updates
    opt = RSOptimizer(n_configs=20, 
                      max_iterations=10, 
                      dynamic_update=False, 
                      n_jobs=-1,
                      verbose=1,
                      seed=SEED)
    opt.search(objective=fn, sampler=sampler, lower_is_better=False)
    df1 = opt.get_df()
    del opt

    # Random search with updates
    opt = RSOptimizer(n_configs=20, 
                      max_iterations=10, 
                      dynamic_update=True, 
                      n_jobs=-1,
                      verbose=1,
                      seed=SEED)
    opt.search(objective=fn, sampler=sampler, lower_is_better=False)
    df2 = opt.get_df()

    # Boxplot and swarmplot
    f, ax = plt.subplots(2, 2, sharex=True, sharey=True)    
    mu1 = df1.groupby('iteration')['metric'].mean()
    sns.boxplot(x='iteration', y='metric', data=df1, ax=ax[0, 0])
    sns.swarmplot(x='iteration', y='metric', data=df1, color=".25", ax=ax[0, 0])
    ax[0, 1].plot(range(10), mu1)

    mu2 = df2.groupby('iteration')['metric'].mean()
    sns.boxplot(x='iteration', y='metric', data=df2, ax=ax[1, 0])
    sns.swarmplot(x='iteration', y='metric', data=df2, color=".25", ax=ax[1, 0])
    ax[1, 1].plot(range(10), mu2)

    # Decorate plots
    ax[0, 0].set(xlabel="Iteration", ylabel="Accuracy", title="Dynamic Update=False")
    ax[0, 1].set(xlabel="Iteration", title="Mean Accuracy by Iteration")
    ax[1, 0].set(xlabel="Iteration", ylabel="Accuracy", title="Dynamic Update=True")
    ax[1, 1].set(xlabel="Iteration", title="Mean Accuracy by Iteration")
    plt.tight_layout()
    plt.show()

    # Make plot

    import pdb; pdb.set_trace()

def main():
    """Creates figures for MODSIM World 2020 paper.
    """

    rs_vs_rsdu()


if __name__ == "__main__":
    main()