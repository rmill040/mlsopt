import logging
import numpy as np

# Package imports
from ..base.optimizers import BaseOptimizer

__all__ = ["GAOptimizer"]

_LOGGER = logging.getLogger(__name__)


class GAOptimizer(BaseOptimizer):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
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
                 backend='loky',
                 verbose=0
                 ):
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
        return ""

    def __repr__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return ""

    def _init_population(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        import pdb; pdb.set_trace()

    def search(self, objective, sampler, lower_is_better):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self._init_population()