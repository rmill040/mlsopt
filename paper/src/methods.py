from functools import partial
import hyperopt
from joblib import delayed, Parallel
import logging
from multiprocessing import Manager
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
    'sgbm' : {
        'n_estimators'  : [100, 500, 1000],
        'max_depth'     : [1, 6, 11],
        'subsample'     : [0.8, 1.0],
        'max_features'  : [0.8, 1.0],
        'learning_rate' : [0.1, 0.01],
        'ccp_alpha'     : [0.0, 0.01, 0.02]
        },
    'xgb'  : {
        'n_estimators'     : [100, 500, 1000],
        'max_depth'        : [1, 6, 11],
        'subsample'        : [0.8, 1.0],
        'colsample_bytree' : [0.8, 1.0],
        'learning_rate'    : [0.1, 0.01],
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


def tpe_optimizer(X, y, name, model, scoring, max_configs, rng):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # To numpy
    X          = X.values
    y          = y.values
    n_features = X.shape[1]
    
    # Define samplers
    feature_sampler = BernoulliFeatureSampler(n_features=n_features)
    hp_sampler      = HP_SAMPLERS[model]()
    space           = {**feature_sampler.space, **hp_sampler.space}
      
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
        X_        = X[:, cols2keep]
               
        # Define model
        if model == 'xgb':
            space['n_jobs'] = -1
            n_jobs          = 1
        else:
            n_jobs = 5
    
        clf = MODELS[model](**space, random_state=SEED)
        
        # Run cross-validation
        metric = CV(clf, X_, y, scoring=scoring, n_jobs=n_jobs).mean()

        return {
            'loss'   : 1 - metric,
            'status' : hyperopt.STATUS_OK,
        }
    
    # Define optimizer and start timer
    trials = hyperopt.Trials()
    
    # Start timer
    _LOGGER.info(f"starting tpe estimator with {max_configs} iterations")
    start = time.time()
    hyperopt.fmin(fn=objective,
                  space=space,
                  algo=hyperopt.tpe.suggest,
                  trials=trials,
                  max_evals=max_configs,
                  rstate=rng)
    
    # Total time   
    minutes = (time.time() - start) / 60
    
    return {
        'method' : 'tpe',
        'metric' : 1 - trials.best_trial['result']['loss'],
        'idx'    : pd.DataFrame(trials.results)['loss'].idxmin() + 1,
        'time'   : minutes
    }


def fs_with_grid_search(X, y, name, model, scoring, max_configs):
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
    best_results           = Manager().dict()
    best_results['metric'] = -np.inf
    best_results['idx']    = None
    n_configs              = len(grid)
    
    def objective(config, i, n_configs):
        """ADD HERE.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if i % 20 == 0:
            _LOGGER.info(f"fs w/ gs iteration {i}/{n_configs}")
        
        # Update pipeline classifier parameters
        pipe.named_steps['clf'].set_params(**config)
        
        # Run cross-validation
        metric = CV(pipe, X, y, scoring=scoring, n_jobs=1).mean()
        if metric > best_results['metric']:
            best_results['metric'] = metric
            best_results['idx']    = i
            _LOGGER.info(f"new best metric {best_results['metric']}, iteration = {i}")
        
        return metric
        
    # Start timer
    start = time.time()
        
    # Calculate metrics for configs
    _LOGGER.info(f"starting fs w/ grid search with {n_configs} configurations")
    Parallel(n_jobs=-1, verbose=False, backend='loky')\
        (delayed(objective)
            (config, i, n_configs)
                for i, config in enumerate(grid, 1))

    # Total time
    minutes = (time.time() - start) / 60
    
    return {
        'method' : 'fs_with_gs',
        'metric' : best_results['metric'],
        'idx'    : best_results['idx'],
        'time'   : minutes
    }


def rs_optimizer(X, y, name, model, scoring, max_configs, dynamic_update):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # To numpy
    X          = X.values
    y          = y.values
    n_features = X.shape[1]

    # Define samplers
    feature_sampler = BernoulliFeatureSampler(n_features=n_features,
                                              muting_threshold=0.25,
                                              dynamic_update=dynamic_update)
    hp_sampler      = HP_SAMPLERS[model](dynamic_update=dynamic_update,
                                         early_stopping=False)
    sampler         = PipelineSampler(seed=SEED)\
                        .register_sampler(feature_sampler, name='feature')\
                        .register_sampler(hp_sampler, name='hp')

    def objective(params): 
        """Objective function to maximize using random search.

        Parameters
        ----------
        params : dict
            Sample from the distributions of features and hyperparameters.

        Returns
        -------
        dict
            Results of parameter evaluation.
        """
        try:
            # Subset features
            cols2keep = params['feature'] # Matches name of feature_sampler
            X_        = X[:, cols2keep]
            
            # Define model
            hp  = params['hp']       # Matches name of hp_sampler
            clf = MODELS[model](**hp, random_state=SEED)
            
            # Run 5-fold stratified cross-validation using AUC as metric
            metric = CV(clf, X_, y, scoring=scoring, n_jobs=1).mean()
            
            return {
                'status'  : utils.STATUS_OK,
                'metric'  : metric,
                'message' : None
            }
        
        except Exception as e:
            return {
                'status'  : utils.STATUS_FAIL,
                'metric'  : 0.0,
                'message' : e
            }

    # Start timer
    start = time.time()
    
    # Define optimizer
    opt = RSOptimizer(n_configs=int(max_configs/5),
                      max_iterations=5,
                      subsample_factor=2,
                      dynamic_update=dynamic_update,
                      verbose=1, 
                      n_jobs=-1,
                      backend='loky',
                      seed=SEED)

    opt.search(objective=objective, 
               sampler=sampler, 
               lower_is_better=False)

    # Total time
    minutes = (time.time() - start) / 60

    # Find best result
    df  = pd.concat([pd.DataFrame(h) for h in opt.history], axis=0)\
            .reset_index(drop=True)
    idx = df['metric'].idxmax()
    
    return {
        'method' : 'rsdu' if dynamic_update else 'rs',
        'metric' : df.loc[idx, 'metric'],
        'idx'    : idx + 1,
        'time'   : minutes
    }


def ga_optimizer(X, y, name, model, scoring, max_configs):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # To numpy
    X          = X.values
    y          = y.values
    n_features = X.shape[1]

    # Define samplers
    feature_sampler = BernoulliFeatureSampler(n_features=n_features)
    hp_sampler      = HP_SAMPLERS[model]()
    sampler         = PipelineSampler(seed=SEED)\
                        .register_sampler(feature_sampler, name='feature')\
                        .register_sampler(hp_sampler, name='hp')

    def objective(params): 
        """Objective function to maximize using ADD HERE.

        Parameters
        ----------
        params : dict
            Sample from the distributions of features and hyperparameters.
        
        Returns
        -------
        dict
            Results of parameter evaluation.
        """
        try:
            # Subset features
            cols2keep = params['feature'] # Matches name of feature_sampler
            X_        = X[:, cols2keep]
            
            # Define model
            hp  = params['hp']       # Matches name of hp_sampler
            clf = MODELS[model](**hp, random_state=SEED)
            
            # Run 5-fold stratified cross-validation using AUC as metric
            metric = CV(clf, X_, y, scoring=scoring, n_jobs=1).mean()
            
            return {
                'status'  : utils.STATUS_OK,
                'metric'  : metric,
                'message' : None
            }
        
        except Exception as e:
            return {
                'status'  : utils.STATUS_FAIL,
                'metric'  : 0.0,
                'message' : e
            }

    # Start timer
    start = time.time()
    
    # Define optimizer
    opt = GAOptimizer(n_population=int(max_configs/5), 
                      n_generations=5,
                      crossover_proba=0.50,
                      mutation_proba=0.10,
                      crossover_independent_proba=0.20,
                      mutation_independent_proba=0.05,
                      tournament_size=3,
                      n_hof=1,
                      n_generations_patience=2, 
                      verbose=1, 
                      n_jobs=-1,
                      backend='loky',
                      seed=SEED)

    opt.search(objective=objective, 
               sampler=sampler, 
               lower_is_better=False)

    # Total time
    minutes = (time.time() - start) / 60

    # Find best result
    df  = pd.concat([pd.DataFrame(h) for h in opt.history], axis=0)\
            .reset_index(drop=True)
    idx = df['metric'].idxmax()
    
    return {
        'method' : 'ga',
        'metric' : df.loc[idx, 'metric'],
        'idx'    : idx + 1,
        'time'   : minutes
    }


def ps_optimizer(X, y, name, model, scoring, max_configs):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # To numpy
    X          = X.values
    y          = y.values
    n_features = X.shape[1]

    # Define samplers
    feature_sampler = BernoulliFeatureSampler(n_features=n_features)
    hp_sampler      = HP_SAMPLERS[model]()
    sampler         = PipelineSampler(seed=SEED)\
                        .register_sampler(feature_sampler, name='feature')\
                        .register_sampler(hp_sampler, name='hp')

    def objective(params): 
        """Objective function to maximize using ADD HERE.

        Parameters
        ----------
        params : dict
            Sample from the distributions of features and hyperparameters.

        Returns
        -------
        dict
            Results of parameter evaluation.
        """
        try:
            # Subset features
            cols2keep = params['feature'] # Matches name of feature_sampler
            X_        = X[:, cols2keep]
            
            # Define model
            hp  = params['hp']       # Matches name of hp_sampler
            clf = MODELS[model](**hp, random_state=SEED)
            
            # Run 5-fold stratified cross-validation using AUC as metric
            metric = CV(clf, X_, y, scoring=scoring, n_jobs=1).mean()
            
            return {
                'status'  : utils.STATUS_OK,
                'metric'  : metric,
                'message' : None
            }
        
        except Exception as e:
            return {
                'status'  : utils.STATUS_FAIL,
                'metric'  : 0.0,
                'message' : e
            }

    # Start timer
    start = time.time()
    
    # Define optimizer
    opt = PSOptimizer(n_particles=int(max_configs/5),
                      max_iterations=5,
                      omega_bounds=(0.10, 0.90),
                      v_bounds=None,
                      phi_p=1.0,
                      phi_g=2.0,
                      tolerance=1e-6,
                      verbose=1, 
                      n_jobs=-1,
                      backend='loky',
                      seed=SEED)

    opt.search(objective=objective, 
               sampler=sampler, 
               lower_is_better=False)

    # Total time
    minutes = (time.time() - start) / 60

    # Find best result
    df  = pd.concat([pd.DataFrame(h) for h in opt.history], axis=0)\
            .reset_index(drop=True)
    idx = df['metric'].idxmax()
    
    return {
        'method' : 'pso',
        'metric' : df.loc[idx, 'metric'],
        'idx'    : idx + 1,
        'time'   : minutes
    }
