import numpy as np
from os.path import join
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification

# Constants
MAIN_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = join(MAIN_DIR, 'data')
SEED        = 1718


def simulate_data(save_name, n_samples, n_features, n_informative):
    """Simulate classification data sets and save to disk
    
    Parameters
    ----------
    save_name : str
        Name of file to use when saving data.
        
    n_samples : int
        Number of samples.
        
    n_features : int
        Number of features.
        
    n_informative : int
        Number of informative features.
    
    Returns
    -------
    None
    """
    # Generate data
    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_informative=n_informative,
                               n_redundant=0,
                               n_repeated=0,
                               class_sep=1.0,
                               random_state=SEED)

    # Save to data folder
    pd.DataFrame(np.column_stack([X, y]))\
      .to_csv(join(DATA_DIR, save_name), index=False)


def main():
    """Simulate classification data.
    """
    # Data 1
    simulate_data(save_name='sim_n100_p200_i2.csv',
                  n_samples=100,
                  n_features=200,
                  n_informative=2)

    # Data 2
    simulate_data(save_name='sim_n100_p500_i5.csv',
                  n_samples=100,
                  n_features=500,
                  n_informative=5)

    # Data 3
    simulate_data(save_name='sim_n100_p200_i20.csv',
                  n_samples=100,
                  n_features=200,
                  n_informative=20)

    # Data 4
    simulate_data(save_name='sim_n100_p500_i50.csv',
                  n_samples=100,
                  n_features=500,
                  n_informative=50)


if __name__ == "__main__":
    main()