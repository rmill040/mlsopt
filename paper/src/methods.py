from functools import partial
import hyperopt
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
import time
from xgboost import XGBClassifier

# Package imports
from mlsopt import utils
from mlsopt.optimizers import GAOptimizer, PSOptimizer, RSOptimizer
from mlsopt.samplers import (BernoulliFeatureSampler, PipelineSampler,
                             SGBMClassifierSampler, XGBClassifierSampler)

# Constants
SEED        = 1718
MODELS      = {
    'sgbm' : GradientBoostingClassifier,
    'xgb'  : XGBClassifier
}
HP_GRID     = {
    'sgbm' : {},
    'xgb'  : {
        'n_estimators'     : [100, 500, 1000],
        'max_depth'        : [1, 6, 12],
        'subsample'        : [.8, 1.0],
        'colsample_bytree' : [.8, 1.0],
        'learning_rate'    : [0.1, .01],
        'base_score'       : [0.2, 0.5, 0.8]
    }
}
HP_SAMPLERS = {
    'sgbm' : SGBMClassifierSampler,
    'xgb'  : XGBClassifierSampler
}
_SKF        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
CV          = partial(cross_val_score, cv=_SKF)

# Define logger
_LOGGER = logging.getLogger(__name__)


def tpe_optimizer(X, y, name, model, scoring, add_noise):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_features = X.shape[1]
    
    # Define samplers
    _p              = n_features/2 if add_noise else n_features
    feature_sampler = BernoulliFeatureSampler(n_features=_p,
                                              feature_names=X.columns[:_p])
    hp_sampler      = HP_SAMPLERS[model]()
    space           = {**feature_sampler.space, **hp_sampler.space}
   
    # To numpy
    X = X.values
    y = y.values
   
    def objective(space): 
        """Objective function to minimize (since hyperopt minimizes by default).

        Parameters
        ----------
        params : dict
            Sample from the distributions of features and hyperparameters.
        
        Returns
        -------
        dict
            Results of params evaluation.
        """
        # Subset features
        cols2keep = [space.pop(col) for col in feature_sampler.feature_names]
        if add_noise: cols2keep *= 2
        X_ = X[:, cols2keep]
               
        # Define model
        clf = MODELS[model](**space, random_state=SEED)
        
        # Run cross-validation
        metrics = CV(clf, X_, y, scoring=scoring, n_jobs=5)

        return {
            'loss'  : 1 - metrics.mean(),
            'status'  : hyperopt.STATUS_OK,
        }
        
    
    # Define optimizer and start timer
    trials = hyperopt.Trials()
    
    # Start timer
    start = time.time()
    hyperopt.fmin(fn=objective,
                  space=space,
                  algo=hyperopt.tpe.suggest,
                  trials=trials,
                  max_evals=200)
    
    # Total time
    minutes = (time.time() - start) / 60
    
    return {
        'method' : 'tpe',
        'metric' : 1 - trials.best_trial['result']['loss'],
        'time'   : minutes
    }


def fs_with_grid_search(X, y, name, model, scoring, add_noise):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # To numpy
    X = X.values
    y = y.values

    # Define pipeline using random forests as feature selector prior to grid 
    # search
    grid  = list(ParameterGrid(HP_GRID[model]))
    model = MODELS[model](random_state=SEED)
    fs    = SelectFromModel(
                RandomForestClassifier(n_estimators=50, 
                                       max_depth=7,
                                       class_weight="balanced",
                                       random_state=SEED)
                )
    pipe  = Pipeline([('fs', fs), ('clf', model)])

    # Initial best metric
    best_metric = -np.inf

    # Start timer
    start     = time.time()
    n_configs = len(grid) 
    
    def objective(config, i, n_configs):
        #TODO: START HERE
        pass
        
    
    for n, config in enumerate(grid, 1):
        
        if n % 20 == 0:
            _LOGGER.info(f"fs w/ gs iteration {n}/{n_configs}")
        
        # Update pipeline classifier parameters
        pipe.named_steps['clf'].set_params(**config)
        
        # Run cross-validation
        metrics = CV(pipe, X, y, n_jobs=5, scoring=scoring)
        if metrics.mean() > best_metric:
            best_metric = metrics.mean()
            _LOGGER.info(f"new best metric {best_metric}, iteration = {n}")
    
    # Total time
    minutes = (time.time() - start) / 60
    
    return {
        'method' : 'fs_with_gs',
        'metric' : best_metric,
        'time'   : minutes
    }


def rs_optimizer(X, y, name, model, scoring, add_noise):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes  = len(np.unique(y))
    n_samples  = X.shape[0]
    n_features = X.shape[1]

    def objective(space):
        """ADD HERE.
        """
        pass


def rsdu_optimizer(X, y, name, model, scoring, add_noise):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes  = len(np.unique(y))
    n_samples  = X.shape[0]
    n_features = X.shape[1]

    def objective(space):
        """ADD HERE.
        """
        pass


def ga_optimizer(X, y, name, model, scoring, add_noise):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes  = len(np.unique(y))
    n_samples  = X.shape[0]
    n_features = X.shape[1]

    def objective(space):
        """ADD HERE.
        """
        pass

def ps_optimizer(X, y, name, model, scoring, add_noise):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes  = len(np.unique(y))
    n_samples  = X.shape[0]
    n_features = X.shape[1]

    def objective(space):
        """ADD HERE.
        """
        pass
