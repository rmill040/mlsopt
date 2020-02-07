from hyperopt import hp
from hyperopt.pyll import scope
import logging
from math import log

# Package imports
from ..base import BaseSampler
from ..utils import parse_hyperopt_param

__all__ = [
    "XGBClassifierSampler",
    "LGBMClassifierSampler",
    "SGBMClassifierSampler"
]

_LOGGER = logging.getLogger(__name__)


class XGBClassifierSampler(BaseSampler):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 dynamic_update=False, 
                 early_stopping=False, 
                 seed=None):
        self.dynamic_update = dynamic_update
        self.early_stopping = early_stopping
        self.seed           = seed
        self.space          = self._init_space()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def _init_space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        subsampling = (log(0.60), log(1.0))
        penalty     = (log(1e-6), log(10))

        return {
        'n_estimators'      : scope.int(hp.quniform('n_estimators', 10, 2000, 1)),
        'max_depth'         : scope.int(hp.quniform('max_depth', 1, 12, 1)),
        'min_child_weight'  : scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'max_delta_step'    : scope.int(hp.quniform('max_delta_step', 0, 3, 1)),
        'learning_rate'     : hp.loguniform('learning_rate', log(1e-4), log(1)),
        'subsample'         : hp.loguniform('subsample', *subsampling),
        'colsample_bytree'  : hp.loguniform('colsample_bytree', *subsampling),
        'colsample_bylevel' : hp.loguniform('colsample_bylevel', *subsampling),
        'colsample_bynode'  : hp.loguniform('colsample_bynode', *subsampling),
        'gamma'             : hp.loguniform('gamma', *penalty),
        'reg_alpha'         : hp.loguniform('reg_alpha', *penalty),
        'reg_lambda'        : hp.loguniform('reg_lambda', *penalty),
        'base_score'        : hp.loguniform('base_score', log(0.01), log(0.99)),
        'scale_pos_weight'  : hp.loguniform('scale_pos_weight', log(0.1), log(10)),
        'random_state'      : hp.choice('random_state', [self.seed])
        }

    def sample_space(self):
        pass

    def update_space(self, data=None):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self.dynamic_update: return

        # Update search distributions
        for param in self.space.keys():

            # Min and max values of current parameters
            ptype, bounds = parse_hyperopt_param(str(self.space[param]))
            print(param, ptype, bounds)

        import pdb; pdb.set_trace()


class LGBMClassifierSampler:
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 dynamic_update=False, 
                 early_stopping=False, 
                 seed=None):
        self.dynamic_update = dynamic_update
        self.early_stopping = early_stopping
        self.seed           = seed

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @property
    def space(self):
        pass

    def sample_space(self):
        pass

    def update_space(self):
        pass

class SGBMClassifierSampler:
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 dynamic_update=False, 
                 early_stopping=False, 
                 seed=None):
        self.dynamic_update = dynamic_update
        self.early_stopping = early_stopping
        self.seed           = seed

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @property
    def space(self):
        pass

    def sample_space(self):
        pass

    def update_space(self):
        pass