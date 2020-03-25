from typing import Union


# Status indicators for optimizers
STATUS_FAIL = 'FAIL'
STATUS_OK   = 'OK'

# Types of samplers
HP_SAMPLER       = "hyperparameter"
FEATURE_SAMPLER  = "feature"
PIPELINE_SAMPLER = "pipeline"

# Typing aliases
T_NUMERIC = Union[int, float]

# Continuous distributions in ConfigSpace
C_DISTRIBUTIONS = ['NormalFloatHyperparameter'
                   'NormalIntegerHyperparameter',
                   'UniformFloatHyperparameter',
                   'UniformIntegerHyperparameter']

# Discrete distributions in ConfigSpace
D_DISTRIBUTIONS = ['CategoricalHyperparameter',
                   'OrdinalHyperparameter']

# Integer distributions in ConfigSpace
I_DISTRIBUTIONS = ['NormalIntegerHyperparameter',
                   'UniformIntegerHyperparameter']

# Float distributions in ConfigSpace
F_DISTRIBUTIONS = ['NormalFloatHyperparameter',
                   'UniformFloatHyperparameter']