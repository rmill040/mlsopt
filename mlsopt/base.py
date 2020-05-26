from abc import ABC, abstractmethod
import logging
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Manager
import numpy as np
from numpy.random import RandomState
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from typing import Union

# Package imports
from .constants import C_DISTRIBUTIONS, HP_SAMPLER, STATUS_FAIL, STATUS_OK

matplotlib.style.use("ggplot")

__all__ = ["BaseOptimizer", "BaseSampler", "HPSamplerMixin"]
_LOGGER = logging.getLogger(__name__)

# TODO: Add checks to ensure optimizer has a solution and results to save to disk

class BaseOptimizer(BaseEstimator, ABC):
    """Base optimizer class.
    
    Parameters
    ----------
    backend : str, optional (default='loky')
        ADD HERE.
        
    verbose : bool, optional (default=False)
        ADD HERE.
        
    n_jobs : int, optional (default=1)
        ADD HERE.
        
    seed : int or None, optional (default=None)
        ADD HERE.
    """
    @abstractmethod
    def __init__(self, 
                 backend: str = 'loky', 
                 verbose: bool = False, 
                 n_jobs: int = 1, 
                 seed: Union[int] = None) -> None:
        # Define backend for parallel computation
        if backend not in ['loky', 'threading', 'multiprocessing']:
            msg = f"backend {backend} not a valid argument, use loky, threading, " + \
                  "or multiprocessing"
            _LOGGER.error(msg)
            raise ValueError(msg)
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

    @property
    def __typename__(self):
        return self.__class__.__name__
    
    def _initialize(self, 
                    sampler, 
                    lower_is_better: bool):
        """Initialize data structures and parameters for optimizers.
        
        Parameters
        ----------
        sampler : ADD HERE.
            add here
            
        lower_is_better : bool
            Whether a lower metric is better.
        
        Returns
        -------
        None
        """
        # Cache parameter names
        for sname, sdist in sampler._registered.items():
            if sdist.__type__ == 'feature':
                self._hp_names[sname] = sdist.feature_names
            else:
                self._hp_names[sname] = np.array(sdist.space.get_hyperparameter_names())

        # Set attribute and initialize metric
        self.lower_is_better        = lower_is_better
        self.best_results['metric'] = np.inf if lower_is_better else -np.inf

        if self.verbose:
            _LOGGER.info(f"starting {self.__typename__} with {self.n_jobs} " +
                         f"jobs using {self.backend} backend")
    
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
            _LOGGER.info(
                f"evaluating configurations, {self.n_configurations - i} remaining"
                )
    
        # Evaluate configuration
        results = objective(configuration)

        # Check if failure occurred during objective func evaluation
        if STATUS_FAIL in results['status'].upper() or STATUS_OK not in results['status'].upper():
            msg = "running configuration failed"
            if 'message' in results.keys():
                if results['message']: 
                    msg += f" because {results['message']}"
            _LOGGER.warning(msg)            
            results['metric'] = np.inf if self.lower_is_better else -np.inf
        else:
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

    def _optimal_solution(self):
        """Get optimal metric and solution.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        best_metric : float
            ADD HERE.
            
        best_params : dict
            ADD HERE.
        """
        best_metric = np.inf if self.lower_is_better else -np.inf
        best_params = None
        for results in self.history:
            for config in results:
                improve = config['metric'] < best_metric if self.lower_is_better \
                            else config['metric'] > best_metric
                if improve:
                    best_metric = config['metric']
                    best_params = config['params']
        
        return best_metric, best_params

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
        """Save search results to .csv file.
        
        Parameters
        ----------
        save_name : str
            Name of file.
        
        Returns
        -------
        None
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
        

class BaseSampler(BaseEstimator, ABC):
    """Base sampler class.
    
    Parameters
    ----------
    dynamic_updating : bool, optional (default=False)
        ADD HERE.
        
    seed : int or None, optional (default=None)
        ADD HERE.
    """
    @abstractmethod
    def __init__(self, 
                 dynamic_updating: bool = False, 
                 seed: Union[int] = None):
        self.dynamic_updating = dynamic_updating
        self._seed            = 0 if seed is None else seed
        self._valid_sampler   = True

    @abstractmethod
    def __type__(self):
        pass

    @property
    def __typename__(self):
        return type(self).__name__
    
    @property
    def seed(self):
        """Getter method for seed.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Seed value.
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        """Setter method for seed.
        
        Parameters
        ----------
        value : int
            New seed value.
        
        Returns
        -------
        None
        """
        self._seed = value
        if hasattr(self, "space"): 
            self.space.seed(value)
    
    @abstractmethod
    def sample_space(self):
        pass

    @abstractmethod
    def update_space(self):
        pass
    

class HPSamplerMixin(BaseSampler):
    """Hyperparameter sampler mixin.
    
    Parameters
    ----------
    distributions : list or None, optional (default=None)
        Hyperparameter distributions.
    
    dynamic_updating : bool, optional (default=False)
        Whether to allow space updating.
        
    early_stopping : bool, optional (default=False)
        Whether space is updated using early_stopping. If True, then the 
        hyperparemeter `n_estimators` is not updated.
    
    seed : int or None, optional (default=None)
        Random seed for sampler.
    """
    def __init__(self, 
                 distributions=None,
                 dynamic_updating=False, 
                 early_stopping=False,
                 seed=None):
        self.distributions  = distributions 
        self.early_stopping = early_stopping
    
        super().__init__(dynamic_updating=dynamic_updating, seed=seed)
        
        # Initialize distributions
        self._init_distributions()

    @property
    def __type__(self):
        """Type of sampler.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Sampler type.
        """
        return HP_SAMPLER

    def sample_space(self):
        """Sample hyperparameter distributions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict
            Key/value pairs where the key is the hyperparameter name and the 
            value is the hyperparameter value.
        """
        return self.space.sample_configuration().get_dictionary()

    def update_space(self, data=None):
        """Update hyperparameter distributions.
        
        Parameters
        ----------
        data : pandas DataFrame
            Historical hyperparameter configurations used to update distributions.
        
        Returns
        -------
        None
        """
        if not self.dynamic_updating: return
        
        # Data must be a dataframe
        if not isinstance(data, pd.DataFrame):
            msg = f"data of type = {type(data)}, must be pandas DataFrame"
            _LOGGER.error(msg)
            raise ValueError(msg)
        
        # Update search distributions of hyperparameters
        for name in data.columns:
            
            # Do not update n_estimators distribution if early stopping enabled
            if name == 'n_estimators' and self.early_stopping: continue
            
            # Get information on hyperparameter distribution
            hp        = self.space.get_hyperparameter(name)
            dist_type = hp.__class__.__name__
            
            # Numerical distribution
            if dist_type in C_DISTRIBUTIONS:
                min_value        = data[name].min()
                max_value        = data[name].max()
                hp.lower         = min_value
                hp.upper         = max_value
                hp.default_value = min_value
                hp.meta.update({"updated": True})

            # Categorical distribution
            if dist_type == 'CategoricalHyperparameter':
                pass
                
            # Ordinal distribution
            if dist_type == 'OrdinalHyperparameter':
                pass