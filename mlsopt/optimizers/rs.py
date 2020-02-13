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
                                    objective,
                                    candidate,
                                    n_candidates,
                                    i,
                                    iteration):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose:
            if i % self.n_jobs == 0:
                _LOGGER.info(f"evaluating candidate, {n_candidates - i} " + 
                             "remaining")
        
        # Evaluate chromosome
        results = objective(candidate, iteration)

        # Check if failure occurred during objective func evaluation
        if STATUS_FAIL in results['status'].upper() or STATUS_OK not in results['status'].upper():
            msg = "running candidate failed"
            if 'message' in results.keys():
                if results['message']: msg += f" because {results['message']}"
            _LOGGER.warn(msg)
            return {
                'status'    : results['status'],
                'metric'    : np.inf if self.lower_is_better else -np.inf,
                'params'    : candidate,
                'iteration' : iteration,
                'id'        : i
            }
        
        # Find best metric so far and compare results to see if current result is better
        if self.lower_is_better:
            if results['metric'] < self.best_results['metric']:
                _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = candidate        
        else:
            if results['metric'] > self.best_results['metric']:
                _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = candidate        
 
        return {
            'status'    : results['status'],
            'metric'    : results['metric'],
            'params'    : candidate,
            'iteration' : iteration,
            'id'        : i
            }
        
    def _evaluate(self, objective, candidates, iteration):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        n_candidates = len(candidates)

        # Calculate metrics for candidates
        results = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                    (delayed(self._calculate_single_candidate)
                        (objective, candidate, n_candidates, i, iteration)
                            for i, candidate in enumerate(candidates))

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
               lower_is_better,
               max_configs_per_round,
               subsample_factor=2):
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

        _LOGGER.info(f"starting {self.__typename__} with {self.n_jobs} jobs using " + 
                     f"{self.backend} backend")

        # Begin search
        for iteration, n_configs in enumerate(max_configs_per_round, 1):
            
            if self.verbose:
                msg = "\n" + "*"*40 + f"\niteration {iteration}\n" + "*"*40
                _LOGGER.info(msg)

            # Grab hof candidates for this round
            hof = int(np.ceil(n_configs/subsample_factor))

            # Sample space
            candidates = sampler.sample_space(n_samples=n_configs)

            # Evaluate candidates
            results = self._evaluate(objective=objective,
                                     candidates=candidates,
                                     iteration=iteration)


            # Update search space now
            _LOGGER.info("updating search space")
            self._update_space(sampler, hof)

        # Finished
        minutes = round((time.time() - start) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes")

        return self

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
        df.sort_values(by='metric', ascending=self.lower_is_better)\
          .to_csv(save_name, index=False)
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