from joblib import delayed, Parallel
import logging
from math import exp
import numpy as np
import pandas as pd
import time

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
                 n_configs,
                 max_iterations,
                 subsample_factor=2,
                 dynamic_update=True,
                 n_jobs=1,
                 backend='loky',
                 verbose=0,
                 seed=None):
        self.n_configs        = n_configs
        self.max_iterations   = max_iterations
        self.subsample_factor = subsample_factor
        self.dynamic_update   = dynamic_update

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

    def _calculate_single_config(self,
                                 objective,
                                 config,
                                 i,
                                 iteration):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose and i % self.n_jobs == 0:
            _LOGGER.info(f"evaluating config, {self.n_configs - i} " + 
                         "remaining")
        
        # Evaluate configuration
        results = objective(config)

        # Check if failure occurred during objective func evaluation
        if STATUS_FAIL in results['status'].upper() or STATUS_OK not in results['status'].upper():
            msg = "running config failed"
            if 'message' in results.keys():
                if results['message']: msg += f" because {results['message']}"
            _LOGGER.warn(msg)
            return {
                'status'    : results['status'],
                'metric'    : np.inf if self.lower_is_better else -np.inf,
                'params'    : config,
                'iteration' : iteration,
                'id'        : i
            }
        
        # Find best metric so far and compare results to see if current result is better
        if self.lower_is_better:
            if results['metric'] < self.best_results['metric']:
                if self.verbose:
                    _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = config        
        else:
            if results['metric'] > self.best_results['metric']:
                if self.verbose:
                    _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = config        
 
        return {
            'status'    : results['status'],
            'metric'    : results['metric'],
            'params'    : config,
            'iteration' : iteration,
            'id'        : i
            }
        
    def _evaluate(self, objective, configs, iteration):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        # Calculate metrics for configs
        results = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                    (delayed(self._calculate_single_config)
                        (objective, config, i, iteration)
                            for i, config in enumerate(configs))

        # Sort population based on metrics
        results = sorted(results, 
                         key=lambda x: x['metric'], 
                         reverse=~self.lower_is_better)

        # Add to history
        self.history.append(results)

        return results

    def _update_space(self, sampler, hof):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self.dynamic_update: return
        # Sort by best metrics
        df = pd.DataFrame(self.history[-1])\
               .sort_values(by='metric', ascending=self.lower_is_better)[:hof]

        # Update sampler spaces
        for sname in sampler.samplers.keys():
            df_space = pd.DataFrame.from_records(
                df['params'].apply(lambda x: x[sname]).values.tolist()
            )
            sampler.update_space(data=df_space, name=sname)


    def search(self,
               objective, 
               sampler, 
               lower_is_better):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        start = time.time()
        
        # Get feature names if specified in sampler
        self._colmapper = {}
        for sname, sdist in sampler.samplers.items():
            if sdist.__type__ == 'feature':
                self._colmapper[sname] = sdist.feature_names
  
        # Set attributes and best results
        self.lower_is_better        = lower_is_better
        self.best_results['metric'] = np.inf if lower_is_better else -np.inf
        
        if self.verbose:
            _LOGGER.info(f"starting {self.__typename__} with {self.n_jobs} " +
                         f"jobs using {self.backend} backend")

        # Begin search
        for iteration in range(1, self.max_iterations + 1):
            
            if self.verbose:
                msg = "\n" + "*"*40 + f"\niteration {iteration}\n" + "*"*40
                _LOGGER.info(msg)

            # Grab hof configs for this round
            hof = int(np.ceil(self.n_configs/self.subsample_factor))

            # Sample space
            configs = sampler.sample_space(n_samples=self.n_configs)

            # Evaluate configs
            results = self._evaluate(objective=objective,
                                     configs=configs,
                                     iteration=iteration)

            # Update search space now
            if self.dynamic_update:
                if self.verbose:
                    _LOGGER.info("updating search space")
                self._update_space(sampler, hof)

        # Finished
        minutes = round((time.time() - start) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes")

        return self