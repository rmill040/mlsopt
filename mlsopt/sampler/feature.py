from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import logging
import numpy as np

# Package imports
from ..base.sampler import BaseSampler

__all__ = ["FeatureSampler"]

_LOGGER = logging.getLogger(__name__)

# TODO:
# 1. Add error checking
# 2. Add unit tests


class FeatureSampler(BaseSampler):
    """Probabilistic sampler for feature space with options of dynamic updates 
    and feature muting.

    Parameters
    ----------
    n_features : int
        ADD HERE.
    """
    def __init__(self, 
                 n_features, 
                 selection_probs=None,
                 feature_names=None,
                 muting_threshold=0.0,
                 dynamic_update=False):
        self.n_features = n_features

        # Set probability of feature on/off equal
        if selection_probs is None:
            self.selection_probs = np.repeat(0.5, self.n_features)
        
        if feature_names is None:
            self._default_feature_names()
        
        self.muting_threshold = muting_threshold
        self.dynamic_update   = dynamic_update
        self.support          = np.repeat(True, self.n_features)

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return self.__repr__()

    def __repr__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.n_features < 4:
            str_sp = "["  + ",".join(self.selection_probs.astype(str).tolist())
            str_sp = str_sp + "]"

            str_fn = "["  + ",".join(self.feature_names.astype(str).tolist())
            str_fn = str_fn + "]"
        else:
            str_sp  = "[" + ",".join(self.selection_probs.astype(str).tolist()[:1])
            str_sp += ",...," + str(self.selection_probs[-1]) + "]"
        
            str_fn  = "[" + ",".join(self.feature_names.astype(str).tolist()[:1])
            str_fn += ",...," + str(self.feature_names[-1]) + "]"
    
        return f"FeatureSampler(n_features={self.n_features}, " + \
               f"selection_probs={str_sp}, feature_names={str_fn}, " + \
               f"muting_threshold={self.muting_threshold}, " + \
               f"dynamic_update={self.dynamic_update})"

    def _default_feature_names(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self.feature_names = np.array(
            [f"f{i}" for i in range(1, self.n_features + 1)]
        )

    @property
    def space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        space = {}
        for name, prob in zip(self.feature_names, self.selection_probs):
            space[name] = hp.pchoice(name,
                                     [(1 - prob, False), (prob, True)])
        return space

    def sample_space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        selected = np.random.binomial(1, 
                                      self.selection_probs, 
                                      self.n_features)\
                                          .astype(bool)
        if self.dynamic_update:
            selected &= self.support
            if not selected.sum():
                _LOGGER.warn("no features selected in sampled space, " + \
                             "reverting to original space")
                selected = self.support

        return selected

    def update_space(self, data): 
        """Update selection probabilities and feature support.
        
        Parameters
        ----------
        data : 
        
        Returns
        -------
        """
        # If no dynamic updates not specified, no need to do anything here
        if not self.dynamic_update: return
            
        # Update selection probabilities
        self.selection_probs = np.mean(data, axis=0)

        # Update feature support
        if self.muting_threshold > 0:
            new_support = self.selection_probs >= self.muting_threshold
            if not new_support.sum():
                _LOGGER.warn(
                    f"no features above muting threshold={self.muting_threshold} " + \
                    f"keeping previous feature set with {self.support.sum()} " + \
                    "features and ignoring update"
                )
                return

            self.support = new_support
            _LOGGER.info(f"feature sampler updated with {self.support.sum()}/" + \
                        f"{self.n_features} features available")