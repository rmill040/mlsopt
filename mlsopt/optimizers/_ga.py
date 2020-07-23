from copy import deepcopy
from joblib import delayed, Parallel
import logging
import numpy as np
import time

# Package imports
from ..base import BaseOptimizer
from ..constants import FEATURE_SAMPLER, HP_SAMPLER

_LOGGER = logging.getLogger(__name__)


class GAOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer.

    Parameters
    ----------
    n_population : int, optional (default=20)
        Size of population.
        
    n_generations : int, optional (default=10)
        Number of generations.
        
    crossover_proba : float, optional (default=0.50)
        Probability of chromosome crossover.
        
    mutation_proba : float, optional (default=0.10)
        Probability of chromosome mutation.
        
    crossover_independent_proba : float, optional (default=0.20)
        Probability of gene crossover within a chromosome.
        
    mutation_independent_proba : float, optional (default=0.05)
        Probability of gene mutation within a chromosome.
        
    tournament_size : int, optional (default=3)
        Number of chromosomes to include for tournament selection.
        
    n_hof : int, optional (default=1)
        Number of top performing chromosomes to keep in hall of fame.
    
    n_generations_patience : int or None, optional (default=None)
        Number of generations to wait before early stopping optimization.
    
    n_jobs : int, optional (default=1)
        Number of parallel jobs to run.
        
    backend : str, optional (default='loky')
        Backend for parallel processing.
        
    verbose : bool, optional (default=False)
        Controls verbosity of analysis.
        
    seed : int or None, optional (default=0)
        Random seed.
    """
    def __init__(self, 
                 n_population=20,
                 n_generations=10,
                 crossover_proba=0.50,
                 mutation_proba=0.10,
                 crossover_independent_proba=0.20,
                 mutation_independent_proba=0.05,
                 tournament_size=3,
                 n_hof=1,
                 n_generations_patience=None,
                 n_jobs=1,
                 backend='loky',
                 verbose=False,
                 seed=None
                 ):
        # TODO: ADD ERROR CHECKING (EX: POP SIZE, HOF SIZE, TOURNAMENT SIZE)
        # TODO: n_generations_patience should be less than n_generations
        #       or early stopping
        # Define attributes
        self.n_population                = n_population
        self.n_configurations            = n_population  # Used in base class
        
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
        
        self._prev_hof_metrics           = []
        self._wait                       = 0

        super().__init__(backend=backend,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         seed=seed)

    def _fitness(self, population, objective, generation):
        """Fitness function to evaluate strength of population.
        
        Parameters
        ----------
        population : add here
            add here.
            
        objective : add here
            add here.
            
        generation : add here
            add here.
        
        Returns
        -------
        list
            ADD HERE.
        """
        # Calculate fitness for population
        population = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                        (delayed(self._evaluate_single_config)
                            (objective, chromosome, i, generation)
                                for i, chromosome in enumerate(population))
        
        # Sort population based on fitness
        population = sorted(population, 
                            key=lambda x: x['metric'], 
                            reverse=~self.lower_is_better)
        
        # Add to history
        self.history_.append(population)

        return population

    def _selection(self, population, generation):
        """Select parents using tournament selection strategy.
        
        Parameters
        ----------
        population : list
            Chromosomes and corresponding fitness values.

        generation : int
            ADD HERE
        
        Returns
        -------
        parents : list
            Selected parent chromosomes.
        """
        if self.verbose:
            _LOGGER.info("running tournament selection to select new parents")

        parents = []

        # Add hof to parents
        if self.n_hof:
            counter     = 0
            hof_metrics = []
            while counter < self.n_hof:
                hof = population.pop(0)
                hof_metrics.append(hof['metric'])
                parents.append(hof['params'])
                counter += 1
            
            # Check if hof metrics changed this generation
            if generation == 1:
                self._prev_hof_metrics = deepcopy(hof_metrics)
            else:
                if self._prev_hof_metrics == hof_metrics:
                    self._wait += 1
                    if self.verbose:
                        _LOGGER.info("no change in hof detected in new generation")

            # Early stopping enabled, return empty list to indicate stopping
            # occurred
            if self._wait > self.n_generations_patience:
                if self.verbose:
                    _LOGGER.info(f"early stopping in generation {generation} " + 
                                 "since no change in hof metrics across " + 
                                 f"{self.n_generations_patience} generations")
                return []
        
        # Update _prev_hof_metrics
        self._prev_hof_metrics = deepcopy(hof_metrics)

        # Run tournament selection
        metrics = np.array([pop['metric'] for pop in population])
        kwargs  = {
            'a'       : range(self.n_population - self.n_hof),
            'size'    : self.tournament_size,
            'replace' : False
            }
        selector = np.argmin if self.lower_is_better else np.argmax
        while len(parents) < self.n_population:
            # Randomly select k chromosomes
            idx = self.rng.choice(**kwargs)

            # Run tournament selection and keep parent
            winner_idx = idx[selector(metrics[idx])]
            parents.append(
                population[winner_idx]['params']
            )
        
        return parents

    def _crossover(self, parents):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose:
            _LOGGER.info("running uniform crossover to generate new population")

        population = []
        kwargs     = {
            'a'       : range(self.n_population),
            'size'    : 2,
            'replace' : False
            }

        while len(population) < self.n_population:
            # Randomly sample two parents
            p1, p2  = self.rng.choice(**kwargs)
            parent1 = parents[p1]
            parent2 = parents[p2]

            # Generate random number to determine if crossover should be made
            # with selected parents
            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)

            # Update children if crossover selected
            if self.rng.uniform() < self.crossover_proba:
                # sname := sampler name
                for sname in parent1.keys():
                    # Crossover hyperparameter space
                    if isinstance(parent1[sname], dict):
                        # pname := parameter name
                        for pname in parent1[sname].keys():
                            pv1   = parent1[sname][pname] 
                            pv2   = parent2[sname][pname]
                            dtype = type(pv1)
                            if self.rng.uniform() < self.crossover_independent_proba:
                                # Calculate new child values
                                b   = self.rng.uniform()
                                cv1 = b * pv1 + (1 - b) * pv2
                                cv2 = (1 - b) * pv1 + b * pv2

                                # Cast data types if needed
                                if not isinstance(cv1, dtype):
                                    cv1 = dtype(cv1)
                                if not isinstance(cv2, dtype):
                                    cv2 = dtype(cv2)                                

                                # TODO: Keep original distribution or allow
                                # interpolation as it currently works?
      
                    # Crossover feature space
                    elif isinstance(parent1[sname], np.ndarray):
                        n_attr = len(parent1[sname])
                        for i, pv1, pv2 in zip(range(n_attr), parent1[sname], parent2[sname]):
                            # Update child values by swapping parent values
                            if self.rng.uniform() < self.crossover_independent_proba:
                                child1[sname][i] = pv2
                                child2[sname][i] = pv1

            # Keep children
            population.append(child1)
            population.append(child2)
        
        return population

    def _mutation(self, population, sampler):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose:
            _LOGGER.info("running mutation on new population")
        
        # Loop over each chromosome and mutate based on probabilities
        for idx in range(self.n_population):
            if self.rng.uniform() < self.mutation_proba:
                # sname := sampler name
                for sname in population[idx].keys():
                    # Mutate hyperparameter space
                    if sampler.registered_[sname].__type__ == HP_SAMPLER:
                        space = sampler.registered_[sname].space    
                        # pname := parameter name
                        for pname in population[idx][sname].keys():
                            if self.rng.uniform() < self.mutation_independent_proba:
                                population[idx][sname][pname] = \
                                    space.get_hyperparameter(pname).sample(space.random)
                        
                    # Mutate feature space
                    elif sampler.registered_[sname].__type__ == FEATURE_SAMPLER:
                        n_attr = len(population[idx][sname])
                        change = np.where(
                            self.rng.uniform(size=n_attr) < self.mutation_independent_proba
                            )[0]
                        if len(change):
                            population[idx][sname][change] = ~population[idx][sname][change]

        return population

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
        tic = time.time()
        
        # Initialize parameters
        self._initialize(sampler=sampler, lower_is_better=lower_is_better)
        
        # Initialize population
        population = sampler.sample_space(n_samples=self.n_population)
        if self.verbose:
            _LOGGER.info(f"initialized population with {self.n_population} " + 
                         "chromosomes")
        
        # Begin optimization
        generation = 1
        while generation <= self.n_generations:
            if self.verbose:
                msg = "\n" + "*"*40 + f"\ngeneration {generation}\n" + "*"*40
                _LOGGER.info(msg)

            # Step 1. Evaluate fitness
            population = self._fitness(population, objective, generation)
        
            # Step 2. Selection and check for convergence
            parents = self._selection(population, generation)
            if not len(parents): 
                break

            # Step 3. Crossover
            population = self._crossover(parents)

            # Step 4. Mutation
            population = self._mutation(population, sampler)
            
            # Keep searching
            generation += 1
            
        # Finished
        toc     = time.time()
        minutes = round((toc - tic) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes")

        return self.best_solution_