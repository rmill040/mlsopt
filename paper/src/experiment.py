import logging
import numpy as np
import os
from os.path import join
import pandas as pd
from pathlib import Path
import time

# Directory imports
import methods

# Constants
MAIN_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = join(MAIN_DIR, 'data')
SAVE_NAME   = join(join(MAIN_DIR, 'results'), 'experiment1.csv')
DATA_SETS   = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
SEED        = 1718
RNG         = np.random.RandomState(seed=SEED)
MAX_CONFIGS = 200

# Define logger
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] %(levelname)s - %(message)s"
    )
_LOGGER = logging.getLogger(__name__)


def load_data(name, labels='original', add_noise=False):
    """Loads data, preprocesses, and splits into features and labels.
    
    Parameters
    ----------
    name : str
        Name of data set.
    
    labels : str
        Label manipulation, keep original labels or create one-vs-all such that
        the majority class is coded as 1, else 0.
    
    add_noise : bool
        Whether to add a randomly permuted copy of all features to the original 
        features.
        
    Returns
    -------
    X : 2d array-like
        Array of features.
        
    y : 1d array-like
        Array of labels.
    """
    df = pd.read_csv(join(DATA_DIR, name), dtype=float)
    X  = df.iloc[:, :-1]
    y  = df.iloc[:, -1]

    # Define feature names
    X.columns = [f"f{i}" for i in range(1, df.shape[1])]

    # Double feature space by shuffling features
    if add_noise:
        X_s = X.apply(RNG.permutation, axis=0)\
               .add_suffix("_s")
        X   = pd.concat([X, X_s], axis=1)
    
    # Keep ova labels
    if labels == 'ova':
        y_u = np.unique(y)
        if len(y_u) > 2:
            y_max = y.value_counts().idxmax()
            y     = np.where(y == y_max, 1, 0)
    
    return X, y
    

def update_results(*,
                   summary, 
                   model,
                   name,
                   add_noise,
                   labels,
                   scoring,
                   idx,
                   max_configs,
                   n_samples,
                   n_features,
                   n_classes,
                   method, 
                   metric, 
                   time):
    """ADD HERE.
    """
    summary.append({
        'model'       : model,
        'name'        : name,
        'add_noise'   : add_noise,
        'labels'      : labels,
        'scoring'     : scoring,
        'idx'         : idx,
        'max_configs' : max_configs,
        'n_samples'   : n_samples,
        'n_features'  : n_features,
        'n_classes'   : n_classes,
        'method'      : method,
        'metric'      : metric,
        'time'        : time
    })
    

def main():
    """Main function to run classifier experiment"""

    start = time.time()

    # Begin experiments
    summary = []
    for labels in ['original']:
        scoring = 'accuracy' if labels == 'original' else 'roc_auc'
        for model in ['xgb']:
            for name in DATA_SETS:
                for add_noise in [False]: 
                    
                    # Load data
                    X, y = load_data(name, labels=labels, add_noise=add_noise)
                
                    msg  = "\n" + "*"*40 + \
                        f"\ndata: {name}, model: {model}, " + \
                        f"labels: {labels}, add noise: {add_noise}" + \
                        "\n" + "*"*40
                    _LOGGER.info(msg)

                    _LOGGER.info(f"samples = {X.shape[0]}, features = {X.shape[1]}")

                    # Data structure to hold current experimental results
                    current = {
                                'summary'     : summary,
                                'model'       : model,
                                'name'        : name,
                                'labels'      : labels,
                                'scoring'     : scoring,
                                'max_configs' : MAX_CONFIGS,
                                'add_noise'   : add_noise,
                                'n_samples'   : X.shape[0],
                                'n_features'  : X.shape[1],
                                'n_classes'   : len(np.unique(y)),
                            }

                    # Keyword args for methods
                    kwargs = {
                        'X'           : X,
                        'y'           : y,
                        'name'        : name,
                        'model'       : model,
                        'max_configs' : MAX_CONFIGS,
                        'scoring'     : scoring
                    }

                    # 1. Tree of parzen estimators
                    results = methods.tpe_optimizer(**kwargs, rng=RNG)
                    update_results(**current, **results)
                    del results

                    # 2. Feature selection and grid search
                    results = methods.fs_with_grid_search(**kwargs)
                    update_results(**current, **results)
                    del results

                    # 3. Random search (no dynamic updates)
                    results = methods.rs_optimizer(**kwargs, dynamic_update=False)
                    update_results(**current, **results)
                    del results
                    
                    # 4. Random search (with dynamic updates)
                    results = methods.rs_optimizer(**kwargs, dynamic_update=True)
                    update_results(**current, **results)                    
                    del results

                    # 5. Particle swarm optimization
                    results = methods.ps_optimizer(**kwargs)
                    update_results(**current, **results)                    
                    del results

                    # 6. Genetic algorithm
                    results = methods.ga_optimizer(**kwargs)
                    update_results(**current, **results)                   
                    del results

    # Finished
    minutes = round((time.time() - start) / 60., 2)
    _LOGGER.info(f"finished all experiments in {minutes} minutes")

    # Write results to disk
    pd.DataFrame(summary).to_csv(SAVE_NAME, index=False)

if __name__ == '__main__':
    main()