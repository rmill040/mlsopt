from typing import Union


# Status indicators for optimizers
STATUS_FAIL = 'FAIL'
STATUS_OK   = 'OK'

# Types of classes
HP_TYPE       = "hyperparameter"
FEATURE_TYPE  = "feature"
PIPELINE_TYPE = "pipeline"

# Mypy numeric alias
NUMERIC_TYPE = Union[int, float]

# Continuous distributions in ConfigSpace
C_DISTRIBUTIONS = ['NormalFloatHyperparameter'
                   'NormalIntegerHyperparameter',
                   'UniformFloatHyperparameter',
                   'UniformIntegerHyperparameter']

# Discrete distributions in ConfigSpace
D_DISTRIBUTIONS = ['CategoricalHyperparameter',
                   'OrdinalHyperparameter']