from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
import logging
import numpy as np

# Package imports
from ..base import BaseSampler

__all__ = ["BernoulliFeatureSampler"]
_LOGGER = logging.getLogger(__name__)


class BernoulliFeatureSampler(BaseSampler):
    """Probabilistic sampler for feature space with options of dynamic updates 
    and feature muting.

    Parameters
    ----------
    n_features : int
        Number of features in sampler.
        
    selection_proba : iterable, optional (default=None)
        Iterable with ith index denoting the probability of selecting feature i.
    
    feature_names : iterable, optional (default=None)
        Name of features.
        
    muting_threshold : float, optional (default=0.0)
        Threshold to use when muting features during space updates.
    
    dynamic_update : bool, optional (default=False)
        Whether to allow space updating.
    
    seed : int, optional (default=None)
        Random seed for sampler.
    """
    def __init__(self, 
                 n_features, 
                 selection_proba=None,
                 feature_names=None,
                 muting_threshold=0.0,
                 dynamic_update=False,
                 seed=None):
        self.n_features = n_features

        if selection_proba is None or len(selection_proba) != self.n_features:
            self.selection_proba = np.repeat(0.5, self.n_features)
        
        if feature_names is None or len(feature_names) != self.n_features:
            self._default_feature_names()
        else:
            self.feature_names = feature_names
        
        if not 0.0 <= muting_threshold < 1.0:
            _LOGGER.error(
                "muting_threshold should be in [0, 1), got " + 
                f"{round(muting_threshold, 2)}"
            )
            raise ValueError
    
        self.muting_threshold = muting_threshold
        self.support          = np.repeat(True, self.n_features)
        
        super().__init__(dynamic_update=dynamic_update, seed=seed)
        
        # Initialize distributions
        self._init_distributions()
        
    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.n_features < 4:
            str_sp  = "[" + ",".join(self.selection_proba.astype(str).tolist()) + "]"
            str_fn  = "[" + ",".join(self.feature_names.astype(str).tolist()) + "]"
        else:
            str_sp  = "[" + ",".join(self.selection_proba.astype(str).tolist()[:1])
            str_sp += ",...," + str(self.selection_proba[-1]) + "]"
        
            str_fn  = "[" + ",".join(self.feature_names.astype(str).tolist()[:1])
            str_fn += ",...," + str(self.feature_names[-1]) + "]"
    
        return f"FeatureSampler(n_features={self.n_features}, " + \
               f"selection_proba={str_sp}, feature_names={str_fn}, " + \
               f"muting_threshold={self.muting_threshold}, " + \
               f"dynamic_update={self.dynamic_update}, seed={self.seed})"

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
        n_zfill = len(str(self.n_features))
        self.feature_names = np.array(["f" + f"{i}".zfill(n_zfill)
                                        for i in range(1, self.n_features + 1)])
        
    def _init_distributions(self):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        self.space = ConfigurationSpace(name=self.__typename__, seed=self.seed)
        kwargs     = {
            'choices'       : [False, True],
            'default_value' : True,
            'meta'          : {'support': True, 'updated': False}
            }
        for name, proba in zip(self.feature_names, self.selection_proba):
            name = str(name)
            self.space.add_hyperparameter(
                CSH.CategoricalHyperparameter(name=name, 
                                              weights=[1 - proba, proba],
                                              **kwargs)
            )

    def sample_space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        space = self.space.sample_configuration()
        
        # ConfigSpace sorts keys alphabetically by default, so we update the 
        # sampled values to reflect the ordering provided in self.feature_names
        if not np.all(np.array(space.keys()) == self.feature_names):
            sample = space.get_dictionary()
            values = np.array([True]*self.n_features)
            for i, name in enumerate(self.feature_names):
                values[i] = sample[name]
        else:
            values = self.space.sample_configuration().get_array().astype(bool)

        return values
    
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
        self.selection_proba = np.mean(data, axis=0)
        
        # Update feature support
        if self.muting_threshold > 0:
            new_support = self.selection_proba >= self.muting_threshold
            if True not in new_support:
                _LOGGER.warn(
                    f"no features above muting threshold={self.muting_threshold} " + 
                    f"keeping previous feature set with {self.support.sum()} " + 
                    "features and ignoring update"
                )
            else:
                self.support = new_support
                _LOGGER.info(
                    f"feature sampler updated with {self.support.sum()}/" + 
                    f"{self.n_features} features available"
                )

        # Update space with probabilities and support
        for name, proba, support in zip(self.feature_names, 
                                        self.selection_proba, 
                                        self.support):    
            name = str(name)

            # If support is False, force probabilities to always select 0 and 
            # update meta-data to indicate support is now False
            if not support:
                new_proba = [1, 0]
            else:
                new_proba = [1 - proba, proba]
            self.space.get_hyperparameter(name).probabilities = new_proba

            self.space.get_hyperparameter(name).meta.update({
                "support" : support,
                "updated" : True
                })