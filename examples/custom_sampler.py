from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Package imports
from mlsopt import STATUS_FAIL, STATUS_OK
from mlsopt.optimizers import GAOptimizer
from mlsopt.samplers import (
    BernoulliFeatureSampler, PipelineSampler, XGBClassifierSampler
)

SEED = 1718


def main():
    """ADD HERE.
    """
    pass

if __name__ == "__main__":
    main()