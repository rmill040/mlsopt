from hyperopt import hp
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
import logging
from math import log

# Package imports
from ..base.samplers import BaseSampler
from ..utils import HP_DISTS, parse_hyperopt_param

__all__ = [
    "XGBClassifierSampler",
    "LGBMClassifierSampler",
    "SGBMClassifierSampler"
]

_LOGGER = logging.getLogger(__name__)

# TODO: check if distribution of n_estimators should be qloguniform!!


class XGBClassifierSampler(BaseSampler):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 space=None,
                 dynamic_update=False, 
                 early_stopping=False, 
                 seed=0):        
        self.early_stopping = early_stopping
        self.seed           = seed
        
        if space is None:
            self.space = self._init_space()
        
        super().__init__(dynamic_update=dynamic_update)

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"XGBClassifierSampler(space={self.space.keys()}, dynamic_update=" + \
               f"{self.dynamic_update}, early_stopping={self.early_stopping}," + \
               f"seed={self.seed})"

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
        return "hyperparameter"

    def _init_space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        subsampling = (log(0.40), log(1.0))
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
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return sample(self.space)

    def update_space(self, data=None):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self.dynamic_update: return

        # Update search distributions of hyperparameters
        for param in self.space.keys():
            
            # Do not update n_estimators distribution if early stopping enabled
            if param == 'n_estimators' and self.early_stopping: continue

            # Parse hyperopt distribution and calculate bounds of distribution
            dist_type, bounds = parse_hyperopt_param(str(self.space[param]))
            if dist_type in ['choice', 'pchoice']: continue
            min_value, max_value = data[param].min(), data[param].max()

            # Log transform bounds if log-based distributions
            if "log" in dist_type:
                min_value = log(min_value)
                max_value = log(max_value)

            # Update new distribution
            hp_dist = HP_DISTS[dist_type]

            # For quantile-based distributions, cast sampled value to integer 
            # and add in quantile value for updated distribution
            if dist_type.startswith("q"):
                self.space[param] = scope.int(
                    hp_dist(param, min_value, max_value, bounds[-1])
                )
            else:
                self.space[param] = hp_dist(param, min_value, max_value)


class LGBMClassifierSampler:
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 space=None,
                 dynamic_update=False, 
                 early_stopping=False, 
                 seed=None):
        self.dynamic_update = dynamic_update
        self.early_stopping = early_stopping
        self.seed           = seed
        
        if space is None:
            self.space = self._init_space()
        
        super().__init__()

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"LGBMClassifierSampler(space={self.space.keys()}, dynamic_update=" + \
               f"{self.dynamic_update}, early_stopping={self.early_stopping}," + \
               f"seed={self.seed})"

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
        return "hyperparameter"

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
                 space=None,
                 dynamic_update=False, 
                 early_stopping=False, 
                 seed=None):
        self.dynamic_update = dynamic_update
        self.early_stopping = early_stopping
        self.seed           = seed
        
        if space is None:
            self.space = self._init_space()
        
        super().__init__()

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"SGBMClassifierSampler(space={self.space.keys()}, dynamic_update=" + \
               f"{self.dynamic_update}, early_stopping={self.early_stopping}," + \
               f"seed={self.seed})"

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
        return "hyperparameter"

    def sample_space(self):
        pass

    def update_space(self):
        pass