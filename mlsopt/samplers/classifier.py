from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
import logging
import pandas as pd

# Package imports
from ..base import BaseSampler

__all__ = [
    "LGBMClassifierSampler",
    "SGBMClassifierSampler",
    "XGBClassifierSampler"
]

_LOGGER = logging.getLogger(__name__)


class ClassifierSamplerMixin(BaseSampler):
    """Classifier sampler mixin.
    
    Parameters
    ----------
    distribution : list or None, optional (default=None)
        Hyperparameter distributions.
    
    dynamic_update : bool, optional (default=False)
        Whether to allow space updating.
        
    early_stopping : bool, optional (default=False)
        Whether space is updated using early_stopping. If True, then the 
        hyperparemeter `n_estimators` is not updated.
    
    seed : int, optional (default=None)
        Random seed for sampler.
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
        """String representation of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Class string.
        """
        names = self.space.get_hyperparameter_names()
        return f"{self.__typename__}(space={names}, " + \
               f"dynamic_update={self.dynamic_update}, " + \
               f"early_stopping={self.early_stopping}, seed={self.seed})"

    def __repr__(self):
        """String representation of class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Class string.
        """
        return self.__str__()

    @property
    def __type__(self):
        """Type of sampler.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        str
            Sampler type.
        """
        return "hyperparameter"

    def sample_space(self):
        """Sample hyperparameter distributions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict
            Key/value pairs where the key is the hyperparameter name and the 
            value is the hyperparameter value.
        """
        return self.space.sample_configuration().get_dictionary()

    def update_space(self, data=None):
        """Update hyperparameter distributions.
        
        Parameters
        ----------
        data : pandas DataFrame
            Historical hyperparameter configurations used to update distributions.
        
        Returns
        -------
        None
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
                hp.meta.update({"updated": True})

            # Categorical distribution
            if dist_type == 'CategoricalHyperparameter':
                pass
                
            # Ordinal distribution
            if dist_type == 'OrdinalHyperparameter':
                pass


class XGBClassifierSampler(ClassifierSamplerMixin):
    """Extreme gradient boosted tree classifier sampler.
    """
    def _init_distributions(self):
        """Initialize hyperparameter distributions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.space = ConfigurationSpace(name=self.__typename__, seed=self.seed)
        meta       = {"updated": False}
        
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
                                               meta=meta)
            ]
        
        # Add hyperparameters now
        self.space.add_hyperparameters(self.distributions) 


class LGBMClassifierSampler(ClassifierSamplerMixin):
    """Light gradient boosting tree classifier sampler.
    """
    def _init_distributions(self):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        raise NotImplementedError("in progress")


class SGBMClassifierSampler(BaseSampler):
    """Scikit-learn gradient boosting tree classifier sampler.
    """
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
                                                 default_value=3,
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
                # Maximum features
                CSH.UniformFloatHyperparameter("max_features",
                                               lower=0.10,
                                               upper=1.0,
                                               log=True,
                                               default_value=0.10,
                                               meta=meta),
                # CCP alpha
                CSH.UniformFloatHyperparameter("ccp_alpha",
                                               lower=1e-4,
                                               upper=0.50,
                                               log=True,
                                               default_value=1,
                                               meta=meta)
            ]
        
        # Add hyperparameters now
        self.space.add_hyperparameters(self.distributions)