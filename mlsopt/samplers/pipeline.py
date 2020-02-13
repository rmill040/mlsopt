import logging
import pandas as pd

# Package imports
from ..base.samplers import BaseSampler

__all__ = ["PipelineSampler"]
_LOGGER = logging.getLogger(__name__)


class PipelineSampler(BaseSampler):
    """ADD
    
    Parameters
    ----------
    """
    def __init__(self, dynamic_update=False):
        self.samplers     = {}
        self._initialized = False

        super().__init__(dynamic_update=dynamic_update)

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"PipelineSampler(dynamic_update={self.dynamic_update})"
    
    def __repr__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return self.__str__()

    @property
    def __type__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
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
            _LOGGER.exception(f"sampler <{name}> not recognized as a valid " + \
                              "sampler")
            raise ValueError

        if name is None:
            _LOGGER.warn("sampler name not defined, registering sampler with " + \
                        f"name <{sampler.__typename__}>")
            name = sampler.__typename__

        # Check if name already exists in registered samplers
        if name in self.samplers.keys():
            _LOGGER.warn(f"sampler <{name}> already registered as a " + \
                         f"<{self.samplers[name].__type__}> sampler, " + \
                         f"replacing sampler with new <{sampler.__type__}> " + \
                         "sampler")
        
        self.samplers[name] = sampler
        self._initialized   = True
        return self

    @property
    def space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self._initialized:
            _LOGGER.exception("sampler not initialized, space undefined")
            raise ValueError
        
        space = {}
        for sampler in self.samplers.values():
            space.update(sampler.space)
        return space

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
            _LOGGER.exception("sampler not initialized, space undefined")
            raise ValueError
        
        if not self.dynamic_update: return

        try:
            self.samplers[name].update_space(data)
        except Exception as e:
            _LOGGER.exception("error trying to update space for sampler " + \
                             f"<{name}> because {e}")
            raise e