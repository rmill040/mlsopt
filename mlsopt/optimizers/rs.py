from joblib import delayed, Parallel
import logging
from math import exp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

matplotlib.style.use("ggplot")

# Package imports
from ..base.optimizers import BaseOptimizer
from ..utils import parse_hyperopt_param, STATUS_FAIL, STATUS_OK

__all__ = ["RSOptimizer"]
_LOGGER = logging.getLogger(__name__)


class RSOptimizer(BaseOptimizer):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 n_jobs=1,
                 backend='loky',
                 verbose=0,
                 seed=None):
        super().__init__(backend=backend,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         seed=seed)   
    
    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def __repr__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def _calculate_single_candidate(self,
                                    objective):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def search(self,
               objective, 
               sampler, 
               lower_is_better,
               subsample_factor=2,
               max_iterations=5):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def serialize(self, save_name):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not save_name.endswith(".csv"): save_name += ".csv"
        
        # Concatenate history together
        df = pd.concat([pd.DataFrame(h) for h in self.history], axis=0)\
               .reset_index(drop=True)

        # Unroll parameters
        df_params = df.pop('params')
        for sname in df_params.iloc[0].keys():
            columns = self._colmapper.get(sname, None)
            records = df_params.apply(lambda x: x[sname]).values.tolist()
            df      = pd.concat([df, 
                                 pd.DataFrame.from_records(records, columns=columns)
                                 ], axis=1)

        # Write data to disk
        df.to_csv(save_name, index=False)
        if self.verbose:
            _LOGGER.info(f"saved results to disk at {save_name}")

    def plot_history(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Concatenate history together
        df = pd.concat([pd.DataFrame(h) for h in self.history], axis=0)\
               .reset_index(drop=True)
        
        # Boxplot and swarmplot
        sns.boxplot(x='iteration', y='metric', data=df)
        sns.swarmplot(x='iteration', y='metric', data=df, color=".25")

        # Decorate plots
        best   = df.iloc[df['metric'].idxmin()] if self.lower_is_better \
                    else df.iloc[df['metric'].idxmax()]
        title  = f"Best metric = {round(best['metric'], 4)} found in " + \
                 f"iteration {best['iteration']}"
        plt.title(title)
        plt.tight_layout()
        plt.show()