import logging
import pandas as pd

# Package imports
from ..base import BaseSampler

__all__ = ["PipelineSampler"]
_LOGGER = logging.getLogger(__name__)


class PipelineSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(self, seed=None):
        self.samplers     = {}
        self._initialized = False
        
        super().__init__(dynamic_update=True, seed=seed)
        
    def __str__(self):
        """String representation of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Class string.
        """
        return f"{self.__typename__}(dynamic_update={self.dynamic_update}, " + \
               f"seed={self.seed})"
    
    def __repr__(self):
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
    def __type__(self):
        """Type of sampler.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Sampler type.
        """
        return "pipeline"

    def register_sampler(self, sampler, name=None):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not hasattr(sampler, "_valid_sampler"):
            _LOGGER.error(f"sampler <{name}> not recognized as a valid " + 
                          "sampler")
            raise ValueError

        if name is None:    
            _LOGGER.warn("sampler name not defined, registering sampler with " + 
                         f"name <{sampler.__typename__}>")
            name = sampler.__typename__

        # Check if name already exists in registered samplers
        if name in self.samplers:
            _LOGGER.warn(f"sampler <{name}> already registered as a " + 
                         f"<{self.samplers[name].__type__}> sampler, " + 
                         f"replacing sampler with new <{sampler.__type__}> " + 
                         "sampler")
        
        # Add sampler and update seed to match seed defined in constructor
        sampler.seed        = self.seed
        self.samplers[name] = sampler
        self._initialized   = True
        return self

    def sample_space(self, n_samples=1, return_combined=True):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self._initialized:
            _LOGGER.exception("sampler not initialized, space undefined")
            raise ValueError
        
        samples = []
        for _ in range(n_samples):
            samples.append(
                {name: sampler.sample_space() for 
                    name, sampler in self.samplers.items()}
            )

        # Return all samples together if specified
        if return_combined:
            return samples
        
        # Otherwise split sampled spaces if specified
        split_samples = {}
        samples       = pd.DataFrame(samples)
        for name in self.samplers.keys():
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
            _LOGGER.error("sampler not initialized, space undefined")
            raise ValueError
        try:
            self.samplers[name].update_space(data)
        except Exception as e:
            _LOGGER.exception("error trying to update space for sampler " + 
                              f"<{name}> because {e}")
            raise e