import logging
from typing import Union

# Package imports
from ..base import BaseSampler
from ..constants import DISTRIBUTIONS, HP_SAMPLER

_LOGGER = logging.getLogger(__name__)


class CustomHPSampler(BaseSampler):
    """Custom hyperparameter sampler.
    
    Parameters
    ----------
    dynamic_updating : bool, optional (default=False)
        ADD HERE.
        
    seed : int or None, optional (default=None)
        ADD HERE.
    """
    def __init__(self,
                 dynamic_updating: bool = False, 
                 seed: Union[int] = None) -> None:
        super().__init__(dynamic_updating=dynamic_updating, seed=seed)
        self.distributions = []
        self._initialized  = False

    def __str__(self) -> str:
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def __repr__(self) -> str:
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return self.__str__()

    def __type__(self) -> str:
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return HP_SAMPLER

    def add_distribution(self, distribution) -> "CustomHPSampler":
        """ADD HERE
        
        TODO: Check for duplicate parameters?
        
        Parameters
        ----------
        
        Returns
        -------
        """
        dist = type(distribution).__name__
        if dist not in DISTRIBUTIONS:
            msg = f"distribution <{dist}> is not a recognized distribution"
            _LOGGER.warning(msg)
        else:
            self.distributions.append(distribution)
            self._initialized = True
        
        return self

    def sample_space(self):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def update_space(self):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass