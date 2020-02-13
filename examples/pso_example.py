from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Package imports
from mlsopt.optimizers import PSOptimizer
from mlsopt.samplers import (
    BernoulliFeatureSampler, PipelineSampler, XGBClassifierSampler
)
from mlsopt.utils import STATUS_FAIL, STATUS_OK

SEED = 1718

def main():
    """Optimize feature selection and hyperparameters for extreme gradient 
    boosted model using particle swarm optimization.
    """
    #############
    # Load Data #
    #############

    X, y = load_breast_cancer(return_X_y=True)

    ###################
    # Define Samplers #
    ###################

    feature_sampler = BernoulliFeatureSampler(n_features=X.shape[1])
    hp_sampler      = XGBClassifierSampler()
    sampler         = PipelineSampler()\
                        .register_sampler(feature_sampler, name='feature')\
                        .register_sampler(hp_sampler, name='hp')

    #############################
    # Define Objective Function #
    #############################

    def objective(params): 
        """Objective function to maximize using particle swarm optimization.

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
            cols = params['feature'] # Matches name of feature_sampler
            X_   = X[:, cols]
            
            # Define model
            hp  = params['hp']       # Matches name of hp_sampler
            clf = XGBClassifier(**hp, seed=SEED)
            
            # Run 5-fold stratified cross-validation using AUC as metric
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

    ####################
    # Define Optimizer #
    ####################

    # Most of these parameters are set to the default, but are explicitly
    # specified for sake of example
    opt = PSOptimizer(n_particles=50,
                      verbose=1, 
                      n_jobs=-1,
                      backend='loky',
                      seed=1718)

    opt.search(objective=objective, 
              sampler=sampler, 
              lower_is_better=False,
              max_iterations=5)
    
    opt.serialize('psoptimizer_results.csv')
    opt.plot_history()

if __name__ == "__main__":
    main()