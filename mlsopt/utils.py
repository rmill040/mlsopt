from typing import Union


class Constants:
    """Helper class to hold constants.
    """
    # Status indicators for optimizers
    STATUS_FAIL = 'FAIL'
    STATUS_OK   = 'OK'

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