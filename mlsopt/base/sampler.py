from abc import ABC, abstractmethod

__all__ = [
    "BaseSampler"
]

class BaseSampler(ABC):
    """Base sampler class.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def sample_space(self):
        pass

    @abstractmethod
    def update_space(self):
        pass