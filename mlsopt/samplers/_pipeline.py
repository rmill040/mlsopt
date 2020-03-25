import logging
import pandas as pd
from typing import Optional, Union

# Package imports
from ..base import BaseSampler
from ..constants import PIPELINE_SAMPLER

_LOGGER = logging.getLogger(__name__)


class PipelineSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(self, seed: Optional[int] = None) -> None:
        self._registered  = {}
        self._initialized = False
        
        super().__init__(dynamic_updating=True, seed=seed)
        
    def __str__(self) -> str:
        """String representation of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Class string.
        """
        return f"{self.__typename__}(dynamic_updating={self.dynamic_updating}, " + \
               f"seed={self.seed})"
    
    def __repr__(self) -> str:
        """String representation of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Class string.
        """
        return self.__str__()

    @property
    def __type__(self) -> str:
        """Type of sampler.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Sampler type.
        """
        return PIPELINE_SAMPLER

    def register_sampler(self, sampler, name=None) -> "PipelineSampler":
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not hasattr(sampler, "_valid_sampler"):
            msg = f"sampler <{name}> not recognized as a valid sampler"
            _LOGGER.error(msg)
            raise ValueError(msg)

        if name is None:    
            _LOGGER.warn("sampler name not defined, registering sampler with " + 
                         f"name <{sampler.__typename__}>")
            name = sampler.__typename__

        # Check if name already exists in registered samplers
        if name in self._registered:
            _LOGGER.warn(f"sampler <{name}> already registered as a " + 
                         f"<{self._registered[name].__type__}> sampler, " + 
                         f"replacing sampler with new <{sampler.__type__}> " + 
                         "sampler")
        
        # Add sampler and update seed to match seed defined in constructor
        sampler.seed           = self.seed
        self._registered[name] = sampler
        self._initialized      = True
        return self

    def sample_space(self, 
                     n_samples: Optional[int] = 1, 
                     return_combined: Optional[bool] = True):
        """Return samples from registered samplers.
        
        Parameters
        ----------
        n_samples : int, optional (default=1)
            Number of samples to generate.
            
        return_combined : bool, optional (default=True)
            Whether to return sampled spaces combined.
        
        Returns
        -------
        ADD HERE.
        """
        if not self._initialized:
            msg = "sampler not initialized, space undefined"
            _LOGGER.error(msg)
            raise ValueError(msg)
        
        samples = []
        for _ in range(n_samples):
            samples.append(
                {name: sampler.sample_space() for 
                    name, sampler in self._registered.items()}
            )

        # Return all samples together if specified
        if return_combined:
            return samples
        
        # Otherwise split sampled spaces if specified
        split_samples = {}
        samples       = pd.DataFrame(samples)
        for name in self._registered.keys():
            split_samples[name] = pd.DataFrame.from_records(samples[name])
        return split_samples

    def update_space(self, data, name):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self._initialized:
            msg = "sampler not initialized, space undefined"
            _LOGGER.error(msg)
            raise ValueError(msg)
        try:
            self._registered[name].update_space(data)
        except Exception as e:
            msg = f"error trying to update space for sampler <{name}> because {e}"
            _LOGGER.exception(msg)
            raise e(msg)