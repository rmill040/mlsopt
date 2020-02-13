import logging
import numpy as np
import os
from os.path import join
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import time

# Constants
DATA_DIR  = join(Path(__file__).resolve().parents[1], 'data')
DATA_SETS = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
SEED      = 1718

# Set up logger
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] %(levelname)s - %(message)s"
    )
_LOGGER = logging.getLogger(__name__)


def load_data(name, labels='original'):
    """Loads data, preprocesses, and splits into features and labels.

    Parameters
    ----------
    name : str
        Name of data set.

    labels : str
        ADD HERE.

    Returns
    -------
    X : 2d array-like
        Array of features.

    y : 1d array-like
        Array of labels.
    """
    df   = pd.read_csv(join(DATA_DIR, name))
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    
    if labels == 'original':
        return X, y
    else:
        y_u = np.unique(y)
        if len(y_u) > 2:
            y_max = y.value_counts().idxmax()
            y     = np.where(y == y_max, 1, 0)
    
    return X.astype(float), y.astype(float)


def main():
    """Main function to run classifier experiment"""

    start = time.time()

    # Begin experiments
    for name in DATA_SETS:    
        _LOGGER.info("\n" + "*"*40 + f"\ndata set {name}\n" + "*"*40)
        X, y = load_data(name, labels='ova')
        _LOGGER.info(f"samples = {X.shape[0]}, features = {X.shape[1]}")

    # Finished
    minutes = round((time.time() - start) / 60, 2)
    _LOGGER.info(f"finished all experiments in {minutes} minutes")

if __name__ == '__main__':
    main()