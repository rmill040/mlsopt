import ConfigSpace.hyperparameters as CSH
import logging
from math import log
import pandas as pd

# Package imports
from ..base.samplers import BaseSampler

__all__ = [
    "LGBMClassifierSampler",
    "SGBMClassifierSampler",
    "XGBClassifierSampler"
]

_LOGGER = logging.getLogger(__name__)

# TODO: check if distribution of n_estimators should be qloguniform!!


class XGBClassifierSampler(BaseSampler):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 distributions=None,
                 dynamic_update=False, 
                 early_stopping=False,
                 seed=None):
        self.distributions  = distributions 
        self.early_stopping = early_stopping
    
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
        names = self.space.get_hyperparameter_names()
        return f"XGBClassifierSampler(space={names}, " + \
               f"dynamic_update={self.dynamic_update}, " + \
               f"early_stopping={self.early_stopping}, seed={self.seed})"

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

    def _init_distributions(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        meta = {"updated": False}
        
        # Set default distributions if None provided
        if self.distributions is None:  
            self.distributions = [
                # Number of estimators
                CSH.UniformIntegerHyperparameter("n_estimators", 
                                                 lower=50, 
                                                 upper=1_000, 
                                                 q=50,
                                                 default_value=100,
                                                 meta=meta),
                # Max depth of trees
                CSH.UniformIntegerHyperparameter("max_depth",
                                                 lower=1,
                                                 upper=11,
                                                 q=1,
                                                 default_value=6,
                                                 meta=meta),
                # Minimum child weight for split
                CSH.UniformIntegerHyperparameter("min_child_weight",
                                                 lower=1,
                                                 upper=20,
                                                 q=1,
                                                 default_value=1,
                                                 meta=meta),
                # Maximum delta step
                CSH.UniformIntegerHyperparameter("max_delta_step",
                                                 lower=0,
                                                 upper=3,
                                                 q=1,
                                                 default_value=0,
                                                 meta=meta),
                # Learning rate
                CSH.UniformFloatHyperparameter("learning_rate",
                                               lower=1e-3,
                                               upper=0.5,
                                               log=True,
                                               default_value=0.1,
                                               meta=meta),
                # Row subsampling
                CSH.UniformFloatHyperparameter("subsample",
                                               lower=0.5,
                                               upper=1.0,
                                               log=True,
                                               default_value=1.0,
                                               meta=meta),
                # Column subsampling by tree
                CSH.UniformFloatHyperparameter("colsample_bytree",
                                               lower=0.5,
                                               upper=1.0,
                                               log=True,
                                               default_value=1.0,
                                               meta=meta),
                # Column subsampling by level
                CSH.UniformFloatHyperparameter("colsample_bylevel",
                                               lower=0.5,
                                               upper=1.0,
                                               log=True,
                                               default_value=1.0,
                                               meta=meta),
                # Column subsampling by node
                CSH.UniformFloatHyperparameter("colsample_bynode",
                                               lower=0.5,
                                               upper=1.0,
                                               log=True,
                                               default_value=1.0,
                                               meta=meta),
                # Gamma
                CSH.UniformFloatHyperparameter("gamma",
                                               lower=1e-4,
                                               upper=5.0,
                                               log=True,
                                               default_value=1e-4,
                                               meta=meta),
                # Regularization alpha
                CSH.UniformFloatHyperparameter("reg_alpha",
                                               lower=1e-4,
                                               upper=1.0,
                                               log=True,
                                               default_value=1e-4,
                                               meta=meta),
                # Regularization lambda
                CSH.UniformFloatHyperparameter("reg_lambda",
                                               lower=1.0,
                                               upper=4.0,
                                               log=True,
                                               default_value=1,
                                               meta=meta),
                # Base score
                CSH.UniformFloatHyperparameter("base_score",
                                               lower=0.01,
                                               upper=0.99,
                                               log=True,
                                               default_value=0.50,
                                               meta=meta),
                # Scale positive weights
                CSH.UniformFloatHyperparameter("scale_pos_weight",
                                               lower=0.1,
                                               upper=10,
                                               log=True,
                                               default_value=1,
                                               meta=meta),
                # Random state
                CSH.Constant("random_state", self.seed, meta=meta)
            ]
        
        # Add hyperparameters now
        self.space.add_hyperparameters(self.distributions) 

    def sample_space(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return self.space.sample_configuration().get_dictionary()

    def update_space(self, data=None):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not self.dynamic_update: return
        
        # Data must be a dataframe
        if not isinstance(data, pd.DataFrame):
            _LOGGER.error(f"data of type = {type(data)}, must be pandas DataFrame")
            raise ValueError
        
        # Update search distributions of hyperparameters
        for name in data.columns:
            
            # Do not update n_estimators distribution if early stopping enabled
            if name == 'n_estimators' and self.early_stopping: continue
            
            # Get information on hyperparameter distribution
            hp        = self.space.get_hyperparameter(name)
            dist_type = type(hp).__name__
            
            # Numerical distribution
            if dist_type in ['UniformIntegerHyperparameter',
                             'NormalIntegerHyperparameter',
                             'UniformFloatHyperparameter',
                             'NormalFloatHyperparameter']:
                min_value, max_value = data[name].min(), data[name].max()
                hp.lower             = min_value
                hp.upper             = max_value
                hp.default_value     = min_value
                hp.meta              = {"updated": True}
            
            # Categorical distribution
            if dist_type == 'CategoricalHyperparameter':
                pass
                
            # Ordinal distribution
            if dist_type == 'OrdinalHyperparameter':
                pass


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
        
        if space is None:
            self.space = self._init_space()
        
        super().__init__(dynamic_update=dynamic_update, seed=seed)

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"LGBMClassifierSampler(space={self.space.keys()}, dynamic_update=" + \
               f"{self.dynamic_update}, early_stopping={self.early_stopping})"

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


class SGBMClassifierSampler(BaseSampler):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 space=None,
                 dynamic_update=False, 
                 early_stopping=False,
                 seed=None):        
        self.early_stopping = early_stopping
        
        if space is None:
            self.space = self._init_space()
        
        super().__init__(dynamic_update=dynamic_update, seed=seed)

    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return f"SGBMClassifierSampler(space={self.space.keys()}, dynamic_update=" + \
               f"{self.dynamic_update}, early_stopping={self.early_stopping}, " + \
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
        subsampling = (log(0.50), log(1.0))

        return {
            'n_estimators'          : scope.int(hp.quniform('n_estimators', 50, 500, 50)),
            'max_depth'             : scope.int(hp.quniform('max_depth', 1, 11, 1)),
            'min_impurity_decrease' : hp.loguniform('min_impurity_decrease', log(1e-4), log(0.20)),
            'learning_rate'         : hp.loguniform('learning_rate', log(1e-3), log(0.5)),
            'subsample'             : hp.loguniform('subsample', *subsampling),
            'max_features'          : hp.loguniform('max_features', *subsampling),
            'ccp_alpha'             : hp.loguniform('ccp_alpha', log(1e-4), log(0.50))
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
