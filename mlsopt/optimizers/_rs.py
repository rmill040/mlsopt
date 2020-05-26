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
                 dynamic_updating=True,
                 n_jobs=1,
                 backend='loky',
                 verbose=0,
                 seed=None):
        self.n_configurations = n_configurations
        self.max_iterations   = max_iterations
        self.subsample_factor = subsample_factor
        self.dynamic_updating = dynamic_updating

        super().__init__(backend=backend,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         seed=seed)
        
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
        self.history_.append(results)

    def _update_space(self, sampler, hof):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self.dynamic_updating: 
            return
    
        # Sort by best metrics
        df = pd.DataFrame(self.history_[-1])\
               .sort_values(by='metric', ascending=self.lower_is_better)[:hof]

        # Update sampler spaces
        for sname in sampler.registered_.keys():
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
        
        # Initialize parameters
        self._initialize(sampler=sampler, lower_is_better=lower_is_better)

        # Begin search
        for iteration in range(1, self.max_iterations + 1):

            if self.verbose:
                _LOGGER.info("\n" + "*"*40 + f"\niteration {iteration}\n" + "*"*40)

            # Sample space
            configs = sampler.sample_space(n_samples=self.n_configurations)

            # Evaluate configs
            self._evaluate(objective=objective, configs=configs, iteration=iteration)

            # Update search space now
            if self.dynamic_updating:
                # Grab hof configs for this round
                hof = int(np.ceil(self.n_configurations/self.subsample_factor))
                if self.verbose:
                    _LOGGER.info(f"updating search space with {hof} configurations")
                self._update_space(sampler, hof)

        # Finished
        minutes = round((time.time() - start) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes")

        return self.best_solution_