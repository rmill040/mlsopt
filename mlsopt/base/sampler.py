from abc import ABC, abstractmethod

__all__ = [
    "BaseSampler"
]


class BaseSampler(ABC):
    """Base sampler class.
    """
    @abstractmethod
    def _init_space(self):
        pass


    @abstractmethod
    def space(self):
        pass


    @abstractmethod
    def sample_space(self):
        pass


    @abstractmethod
    def update_space(self):
        pass