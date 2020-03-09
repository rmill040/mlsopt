from abc import ABC, abstractmethod

__all__ = ["BaseSampler"]


class BaseSampler(ABC):
    """Base sampler class.
    """
    @abstractmethod
    def __init__(self, dynamic_update, seed=None):
        self.dynamic_update = dynamic_update
        self._seed          = 0 if seed is None else seed
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
    def __typename__(self):
        return type(self).__name__
    
    @property
    def seed(self):
        """Getter method for seed.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Seed value.
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        """Setter method for seed.
        
        Parameters
        ----------
        value : int
            New seed value.
        
        Returns
        -------
        None
        """
        self._seed = value
        if hasattr(self, "space"): 
            self.space.seed(value)
    
    @abstractmethod
    def sample_space(self):
        pass

    @abstractmethod
    def update_space(self):
        pass