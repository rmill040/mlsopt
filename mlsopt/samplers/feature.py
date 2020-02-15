from collections import OrderedDict
from hyperopt import hp
import logging
import numpy as np

# Package imports
from ..base.samplers import BaseSampler

__all__ = ["BernoulliFeatureSampler"]
_LOGGER = logging.getLogger(__name__)

# TODO:
# 1. Add error checking
# 2. Add unit tests
# 3. add verbose as argument

class BernoulliFeatureSampler(BaseSampler):
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
                 dynamic_update=False,
                 seed=None):
        self.n_features = n_features

        if selection_probs is None:
            self.selection_probs = np.repeat(0.5, self.n_features)
        
        if feature_names is None:
            self._default_feature_names()
        else:
            self.feature_names = feature_names
        
        self.muting_threshold = muting_threshold
        self.support          = np.repeat(True, self.n_features)

        super().__init__(dynamic_update=dynamic_update, seed=seed)

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.n_features < 4:
            str_sp  = "[" + ",".join(self.selection_probs.astype(str).tolist()) + "]"
            str_fn  = "[" + ",".join(self.feature_names.astype(str).tolist()) + "]"
        else:
            str_sp  = "[" + ",".join(self.selection_probs.astype(str).tolist()[:1])
            str_sp += ",...," + str(self.selection_probs[-1]) + "]"
        
            str_fn  = "[" + ",".join(self.feature_names.astype(str).tolist()[:1])
            str_fn += ",...," + str(self.feature_names[-1]) + "]"
    
        return f"FeatureSampler(n_features={self.n_features}, " + \
               f"selection_probs={str_sp}, feature_names={str_fn}, " + \
               f"muting_threshold={self.muting_threshold}, " + \
               f"dynamic_update={self.dynamic_update}, seed={seed})"

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
        return "feature"

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
        selected = self.rng.binomial(1, 
                                     self.selection_probs, 
                                     self.n_features)\
                                        .astype(bool)
        if self.dynamic_update:
            selected &= self.support
            if not selected.sum():
                _LOGGER.warn("no features selected in sampled space, " + \
                             "reverting to previous space")
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