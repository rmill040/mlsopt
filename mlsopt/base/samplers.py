from abc import ABC, abstractmethod
from ConfigSpace import ConfigurationSpace

__all__ = ["BaseSampler"]


class BaseSampler(ABC):
    """Base sampler class.
    """
    @abstractmethod
    def __init__(self, dynamic_update, seed=None):
        self.dynamic_update = dynamic_update
        self.seed           = 0 if seed is None else seed
        self._valid_sampler = True
        self.space          = ConfigurationSpace(name=self.__typename__, 
                                                 seed=self.seed)
        self._init_distributions()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __type__(self):
        pass

    @property
    def __typename__(self):
        return type(self).__name__
    
    @abstractmethod
    def _init_distributions(self):
        pass

    @abstractmethod
    def sample_space(self):
        pass

    @abstractmethod
    def update_space(self):
        pass