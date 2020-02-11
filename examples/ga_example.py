import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Package imports
from mlsopt.optimizers import GAOptimizer
from mlsopt.samplers import (
    BernoulliFeatureSampler, PipelineSampler, XGBClassifierSampler
)
from mlsopt.utils import STATUS_FAIL, STATUS_OK


def main():
    """ADD HERE.
    """
    # Data
    X, y = load_breast_cancer(return_X_y=True)

    # Define individual samplers
    hp_sampler      = XGBClassifierSampler(seed=1718)
    feature_sampler = BernoulliFeatureSampler(n_features=X.shape[1])

    # Define pipeline sampler
    sampler = PipelineSampler()
    sampler.register_sampler(hp_sampler, name='hp')
    sampler.register_sampler(feature_sampler, name='feature')

    # Define objective function (fitness function)
    def fn(chromosome): 
        """ADD HERE.
        """
        try:
            # Subset features
            cols = chromosome['feature']
            X_   = X[:, cols]
            
            # Define model
            hp  = chromosome['hp']
            clf = XGBClassifier(**hp)
            
            # Calculate cv score
            metric = cross_val_score(clf, X_, y, cv=5, n_jobs=1, scoring='roc_auc')
            
            return {
                'status'  : STATUS_OK,
                'metric'  : metric.mean(),
                'message' : None
            }
        
        except Exception as e:
            return {
                'status'  : STATUS_FAIL,
                'metric'  : 0.0,
                'message' : e
            }


    # Optimize
    opt = GAOptimizer(verbose=2, n_jobs=-1)
    opt.search(
        fitness=fn,
        sampler=sampler,
        lower_is_better=False
    )


if __name__ == "__main__":
    main()