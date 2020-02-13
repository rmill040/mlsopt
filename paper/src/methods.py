from hyperopt import hp, fmin, Trials, STATUS_OK, tpe
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

# Constants
SEED   = 1718
MODELS = {
    'sgbm' : GradientBoostingClassifier,
    'xgb'  : XGBClassifier
}
CV     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tpe_optimizer(X, y, name, model, scoring):
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

def fs_with_grid_search(X, y, name, model, scoring):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    n_classes  = len(np.unique(y))
    n_samples  = X.shape[0]
    n_features = X.shape[1]

def rs_optimizer(X, y, name, model, scoring, dynamic_update=False):
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

def ga_optimizer(X, y, name, model, scoring):
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

def ps_optimizer(X, y, name, model, scoring):
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