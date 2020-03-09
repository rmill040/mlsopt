from abc import ABC, abstractmethod
import logging
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Manager
import numpy as np
from numpy.random import RandomState
import pandas as pd
import seaborn as sns

# Package imports
from ..utils import STATUS_FAIL, STATUS_OK

matplotlib.style.use("ggplot")

__all__ = ["BaseOptimizer"]
_LOGGER = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Base optimizer class.
    """
    @abstractmethod
    def __init__(self, backend, verbose, n_jobs, seed):
        # Define backend for parallel computation
        if backend not in ['loky', 'threading', 'multiprocessing']:
            _LOGGER.error(f"backend {backend} not a valid argument, use " + 
                          "loky, threading, or multiprocessing")
            raise ValueError
        self.backend = backend
 
        # Calculate number of jobs for parallel processing
        max_cpus = cpu_count()
        if n_jobs == 0:
            n_jobs = 1
        elif abs(n_jobs) > max_cpus:
            n_jobs = max_cpus
        else:
            if n_jobs < 0: n_jobs = list(range(1, cpu_count() + 1))[n_jobs]
        self.n_jobs = n_jobs

        self.verbose   = verbose
        self.seed      = seed
        self.rng       = RandomState(seed=seed)
        self.history   = []
        self._hp_names = {}

        # Keep track of best results
        self.best_results           = Manager().dict()
        self.best_results['metric'] = None
        self.best_results['params'] = None

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    def __typename__(self):
        return type(self).__name__
    
    def _cache_hp_names(self, sampler):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        for sname, sdist in sampler.samplers.items():
            if sdist.__type__ == 'feature':
                self._hp_names[sname] = sdist.feature_names
            else:
                self._hp_names[sname] = np.array(sdist.space.get_hyperparameter_names())
    
    def _evaluate_single_config(self, 
                                objective, 
                                configuration, 
                                i, 
                                iteration):
        """Calculate objective function for a single configuration.
        
        Parameters
        ----------
        objective : callable function
            Function to be optimized.
            
        configuration : dict
            Key/value pairs with the sampler name and sampled values.
            
        i : int
            Configuration ID.
            
        iteration : int
            Iteration of optimizer.
        
        Returns
        -------
        dict
            Key/value pairs containing results of evaluating configuration.
        """
        if self.verbose and i % self.n_jobs == 0:
            _LOGGER.info(f"evaluating configurations, {self.n_configurations - i} " + 
                         "remaining")
    
        # Evaluate configuration
        results = objective(configuration)

        # Check if failure occurred during objective func evaluation
        if STATUS_FAIL in results['status'].upper() or STATUS_OK not in results['status'].upper():
            msg = "running configuration failed"
            if 'message' in results.keys():
                if results['message']: msg += f" because {results['message']}"
            _LOGGER.warn(msg)
            return {
                'status'    : results['status'],
                'metric'    : np.inf if self.lower_is_better else -np.inf,
                'params'    : configuration,
                'iteration' : iteration,
                'id'        : i
            }
        
        # Find best metric so far and compare results to see if current result is better
        if self.lower_is_better:
            if results['metric'] < self.best_results['metric']:
                if self.verbose:
                    _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = configuration        
        else:
            if results['metric'] > self.best_results['metric']:
                if self.verbose:
                    _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = configuration        
 
        return {
            'status'    : results['status'],
            'metric'    : results['metric'],
            'params'    : configuration,
            'iteration' : iteration,
            'id'        : i
            }

    @abstractmethod
    def search(self):
        pass

    def get_df(self):
        """ADD HERE.
        
        Parameters
        ----------
        
        Returns
        -------
        
        """
        # Concatenate history together
        df = pd.concat([pd.DataFrame(h) for h in self.history], axis=0)\
               .reset_index(drop=True)

        # Unroll parameters
        df_params = df.pop('params')
        for sname in df_params.iloc[0].keys():
            records = df_params.apply(lambda x: x[sname]).values.tolist()
            df_tmp  = pd.DataFrame.from_records(records)
            
            # Fix columns if found
            columns = self._hp_names.get(sname, None)
            if columns is not None:
                columns = np.array(
                    list(map(lambda x: sname + "-" + x, columns))
                )
            df_tmp.columns = columns
            df             = pd.concat([df, df_tmp], axis=1)
        
        # Sort by metric
        df = df.sort_values(by='metric', ascending=self.lower_is_better)\
               .reset_index(drop=True)

        return df

    def serialize(self, save_name):
        """ADD

        TODO: What happens in the case with different sampler names and the 
              same parameter names (e.g., 2 xgboost samplers with same parameters 
              being sampled) --> handle this.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not save_name.endswith(".csv"): save_name += ".csv"
        
        df = self.get_df()

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
        df = self.get_df()
        
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