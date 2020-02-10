from abc import ABC, abstractmethod
import logging
from multiprocessing import cpu_count

__all__ = [
    "BaseOptimizer"
]

_LOGGER = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Base optimizer class.
    """    
    @abstractmethod
    def __init__(self, backend, verbose, n_jobs):
        # Define backend for parallel computation
        if backend not in ['loky', 'threading', 'multiprocessing']:
            _LOGGER.exception(f"backend {backend} not a valid argument, use " + \
                              "loky, threading, or multiprocessing")
        self.backend = backend
 
        # Calculate number of jobs for parallel processing
        max_cpus = cpu_count()
        if n_jobs == 0:
            n_jobs = 1
        elif abs(n_jobs) > max_cpus:
            n_jobs = max_cpus
        else:
            if n_jobs < 0: n_jobs = list(range(1, cpu_count()+1))[n_jobs]
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.history = []

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def serialize(self):
        pass