from joblib import delayed, Parallel
import logging
import numpy as np
import pandas as pd
import time

# Package imports
from ..base.optimizers import BaseOptimizer
from ..utils.constants import STATUS_FAIL, STATUS_OK

__all__ = ["GAOptimizer"]

_LOGGER = logging.getLogger(__name__)


class GAOptimizer(BaseOptimizer):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 n_population=20,
                 n_generations=10,
                 crossover_proba=0.50,
                 mutation_proba=0.10,
                 crossover_independent_proba=0.10,
                 mutation_independent_proba=0.05,
                 tournament_size=3,
                 n_hof=1,
                 n_generations_patience=None, #TODO: IMPLEMENT THIS
                 n_jobs=1,
                 backend='loky',
                 verbose=0
                 ):
        # TODO: ADD ERROR CHECKING (EX: POP SIZE, HOF SIZE, TOURNAMENT SIZE)
        # TODO: ADD DATA STRUCTURE TO HOLD BEST METRICS FOR n_generations_patience
        # TODO: n_generations_patience should be less than n_generations
        #       or early stopping
        # Define attributes
        self.n_population                = n_population
        self.n_generations               = n_generations
        self.crossover_proba             = crossover_proba
        self.mutation_proba              = mutation_proba
        self.crossover_independent_proba = crossover_independent_proba
        self.mutation_independent_proba  = mutation_independent_proba
        self.tournament_size             = tournament_size
        self.n_hof                       = n_hof
        self.n_generations_patience      = n_generations_patience
        self.n_jobs                      = n_jobs
        self.backend                     = backend
        self.verbose                     = verbose

        super().__init__(
            backend=backend,
            n_jobs=n_jobs,
            verbose=verbose
            )

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"GAOptimizer(n_population={self.n_population}, " + \
               f"n_generations={self.n_generations}, crossover_proba=" + \
               f"{self.crossover_proba}, mutation_proba={self.mutation_proba}, " + \
               f"crossover_independent_proba={self.crossover_independent_proba}, " + \
               f"mutation_independent_proba={self.mutation_independent_proba}, " + \
               f"tournament_size={self.tournament_size}, n_hof={self.n_hof}, " + \
               f"n_generations_patience={self.n_generations_patience},  " + \
               f"n_jobs={self.n_jobs}, backend={self.backend}, verbose={self.verbose})"

    def __repr__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return self.__str__()

    def _check_params(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def _calculate_fitness(self, 
                           fitness, 
                           lower_is_better,
                           chromosome, 
                           i, 
                           generation):
        """Calculate fitness value for chromosome.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose:
            if i % self.n_jobs == 0:
                _LOGGER.info(f"evaluating chromosomes, {self.n_population - i} " + \
                             "remaining")
        
        # Evaluate chromosome
        results = fitness(chromosome)

        # Check if failure occurred during objective func evaluation
        if STATUS_FAIL in results['status'].upper() or STATUS_OK not in results['status'].upper():
            msg = "running candidate failed"
            if 'message' in results.keys():
                if results['message']: msg += f" because {results['message']}"
            _LOGGER.warn(msg)
            return {
                'status'     : results['status'],
                'metric'     : np.inf if lower_is_better else -np.inf,
                'params'     : chromosome,
                'generation' : generation,
                'id'         : i
            }
        
        # Find best metric so far and compare results to see if current result is better
        if lower_is_better:
            if results['metric'] < self.best_results['metric']:
                _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = chromosome        
        else:
            if results['metric'] > self.best_results['metric']:
                _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = chromosome        
 
        return {
            'status'     : results['status'],
            'metric'     : results['metric'],
            'params'     : chromosome,
            'generation' : generation,
            'id'         : i
            }

    def _selection(self, data):
        """Select parents using tournament selection strategy.
        
        Parameters
        ----------
        data : list
            Chromosomes and corresponding fitness values.
        
        Returns
        -------
        parents : list
            Selected parent chromosomes.
        """
        if self.verbose:
            _LOGGER.info("running tournament selection to select new parents")

        selector = np.argmin if self.lower_is_better else np.argmax
        parents  = []

        # Add hof to parents
        if self.n_hof:
            counter = 0
            while counter < self.n_hof:
                hof = data.pop(0)
                parents.append(hof['params'])
                counter += 1

        # Run tournament selection
        metrics = np.array([d['metric'] for d in data])
        while len(parents) < self.n_population:
            # Randomly select k chromosomes
            idx = np.random.choice(range(self.n_population - self.n_hof), 
                                   size=self.tournament_size, 
                                   replace=False)
            
            # Run tournament selection and keep parent
            winner_idx = selector(metrics[idx])
            parents.append(
                data[winner_idx]['params']
            )
        
        return parents

    def _crossover(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def _mutation(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def search(self, fitness, sampler, lower_is_better):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self.lower_is_better = lower_is_better

        _LOGGER.info(f"beginning search with {self.n_jobs} jobs using " + \
                    f"{self.backend} backend")
        start = time.time()

        # Initialize population and set best results
        population                  = sampler.sample_space(n_samples=self.n_population)
        self.best_results['metric'] = np.inf if lower_is_better else -np.inf
        
        # Begin optimization
        for generation in range(1, self.n_generations+1):

            if self.verbose:
                _LOGGER.info(f"starting generation {generation}")

            # Step 1. Evaluate fitness
            results = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                        (delayed(self._calculate_fitness)\
                            (fitness, lower_is_better, chromosome, i, generation)
                                for i, chromosome in enumerate(population)
                            )
            # Sort results
            results = sorted(results, key=lambda x: x['metric'], reverse=~lower_is_better)

            # Keep results
            self.history.append(results)
        
            # Step 2. Selection
            parents = self._selection(results)

            # Step 3. Crossover
            population = self._crossover(parents)

            # Step 4. Mutation
            population = self._mutation(population)
        
        return self

    def serialize(self, save_name):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        print(self.history)