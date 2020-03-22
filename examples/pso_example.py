from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Package imports
from mlsopt.optimizers import PSOptimizer
from mlsopt.samplers import (
    BernoulliFeatureSampler, PipelineSampler, XGBClassifierSampler
)
from mlsopt.constants import STATUS_FAIL, STATUS_OK

SEED = 1718


def main():
    """Optimize feature selection and hyperparameters for extreme gradient 
    boosted model using particle swarm optimization.
    """
    #############
    # Load Data #
    #############

    X, y          = load_breast_cancer(return_X_y=True)
    feature_names = load_breast_cancer()['feature_names']

    ###################
    # Define Samplers #
    ###################

    feature_sampler = BernoulliFeatureSampler(n_features=X.shape[1],
                                              feature_names=feature_names)
    hp_sampler      = XGBClassifierSampler()
    sampler         = PipelineSampler(seed=SEED)\
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
            clf = XGBClassifier(**hp, random_state=SEED)
            
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
                      max_iterations=5,
                      omega_bounds=(0.10, 0.90),
                      v_bounds=None,
                      phi_p=1.0,
                      phi_g=2.0,
                      tolerance=1e-6,
                      verbose=1, 
                      n_jobs=-1,
                      backend='loky',
                      seed=1718)

    opt.search(objective=objective, 
               sampler=sampler, 
               lower_is_better=False)
    
    opt.serialize('psoptimizer_results.csv')
    opt.plot_history()

if __name__ == "__main__":
    main()