from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
import logging

# Package imports
from ..base import HPSamplerMixin

_LOGGER = logging.getLogger(__name__)


class XGBClassifierSampler(HPSamplerMixin):
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


class LGBMClassifierSampler(HPSamplerMixin):
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


class SGBMClassifierSampler(HPSamplerMixin):
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