from abc import ABC, abstractmethod

__all__ = [
    "BaseSampler"
]

class BaseSampler(ABC):
    """Base sampler class.
    """    
    @abstractmethod
    def __init__(self, dynamic_update):
        self.dynamic_update = dynamic_update
        self._valid_sampler = True

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
    def __sampler__(self):
        return type(self).__name__

    @abstractmethod
    def sample_space(self):
        pass

    @abstractmethod
    def update_space(self):
        pass