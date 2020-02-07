import numpy as np


class GAOptimizer:
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 estimator,
                 feature_sampler,
                 hp_sampler,
                 n_population=100,
                 n_generations=10,
                 crossover_proba=0.50,
                 mutation_proba=0.10,
                 crossover_independent_proba=0.10,
                 mutation_independent_proba=0.05,
                 tournament_size=3,
                 n_hof=1,
                 n_generations_patience=None,
                 n_jobs=1,
                 verbose=0
                 ):
        # Define attributes
        self.estimator                   = estimator
        self.feature_sampler             = feature_sampler
        self.hp_sampler                  = hp_sampler
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
        self.verbose                     = verbose

        # ADD HERE
        self._converged = False


    def _init_population(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        features = self.feature_sampler.sample_space(n_samples=self.n_population)
        # self.hp_sampler.sample(n_samples=self.n_population)


    def search(self, X, y):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass