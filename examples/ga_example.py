import pandas as pd
from sklearn.datasets import load_breast_cancer

# Package imports
from mlsopt.samplers import (
    BernoulliFeatureSampler, PipelineSampler, XGBClassifierSampler
)
from mlsopt.optimizers import GAOptimizer


def main():
    """ADD HERE.
    """
    # Data
    X, y = load_breast_cancer(return_X_y=True)

    # Define individual samplers
    hp_sampler      = XGBClassifierSampler(
        dynamic_update=True, 
        early_stopping=True
        )
    feature_sampler = BernoulliFeatureSampler(n_features=X.shape[1])

    # Define pipeline sampler
    sampler = PipelineSampler()
    sampler.register_sampler(hp_sampler)
    sampler.register_sampler(feature_sampler)

    # Define objective function
    def fn(): pass

    # Optimize
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()