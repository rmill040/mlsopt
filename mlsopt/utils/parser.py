import logging
import re

__all__ = ["parse_hyperopt_param"]

_LOGGER = logging.getLogger(__name__)

# Currently supported hyperopt distributions for parsing
_HP_DIST = [
    "switch",
    "categorical",
    "quniform",
    "uniform",
    "loguniform"
]

def parse_hyperopt_param(string):
    """Parses lower and upper bounds from string representation of hyperopt
    parameter.
    
    Parameters
    ----------
    string : str
        String representation of hyperopt parameter.
    
    Returns
    -------
    param_type : str
        Distribution of parameter.
    
    parsed : list
        Lower and upper bounds of parameter.
    """
    param_type = re.findall("|".join(_HP_DIST), string)
    if not len(param_type):
        _LOGGER.exception("unsupported parameter distribution in hyperopt " + \
                         f"string:\n{string}\nSupported distributions are " + \
                         f"{', '.join(_HP_DIST)}\n")       
        raise ValueError
    
    # Grab first element since this is the distribution type
    param_type = param_type[0]
    if param_type in ['switch', 'categorical']: return param_type, []

    # Get all the Literal{} entries
    string = ' '.join(re.findall("Literal{-*\d+\.*\d*}", string))

    # Parse all the numbers within the {} and map to float
    parsed = list(map(float, re.findall("[Literal{}](-*\d+\.*\d*)", string)))
    return param_type, parsed[:2]
