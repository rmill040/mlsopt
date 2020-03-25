from joblib import delayed, Parallel
import logging
import numpy as np
import pandas as pd
import time

# Package imports
from ..base import BaseOptimizer

_LOGGER = logging.getLogger(__name__)


class RSOptimizer(BaseOptimizer):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 n_configurations,
                 max_iterations,
                 subsample_factor=2,
                 dynamic_update=True,
                 n_jobs=1,
                 backend='loky',
                 verbose=0,
                 seed=None):
        self.n_configurations = n_configurations
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
        
    def _evaluate(self, objective, configs, iteration):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        # Calculate metrics for configs
        results = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                    (delayed(self._evaluate_single_config)
                        (objective, config, i, iteration)
                            for i, config in enumerate(configs))

        # Sort population based on metrics
        results = sorted(results, 
                         key=lambda x: x['metric'], 
                         reverse=~self.lower_is_better)

        # Add to history
        self.history.append(results)

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
        
        # Cache hp names
        self._cache_hp_names(sampler)
  
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

            # Sample space
            configs = sampler.sample_space(n_samples=self.n_configurations)

            # Evaluate configs
            self._evaluate(objective=objective, configs=configs, iteration=iteration)

            # Update search space now
            if self.dynamic_update:
                # Grab hof configs for this round
                hof = int(np.ceil(self.n_configurations/self.subsample_factor))
                if self.verbose:
                    _LOGGER.info("updating search space")
                self._update_space(sampler, hof)

        # Finished
        minutes = round((time.time() - start) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes")

        return self._optimal_solution()