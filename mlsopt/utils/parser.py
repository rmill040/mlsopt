import logging
import re

# Package imports
from ..utils import SUPPORTED_HP_DISTS

__all__ = ["parse_hyperopt_param"]

_LOGGER = logging.getLogger(__name__)


def parse_hyperopt_param(string):
    """Parses lower and upper bounds from string representation of hyperopt
    parameter.
    
    Parameters
    ----------
    string : str
        String representation of hyperopt parameter.
    
    Returns
    -------
    dist_type : str
        Distribution of parameter.
    
    parsed : list
        Lower and upper bounds of parameter.
    """
    dist_type = re.findall("|".join(SUPPORTED_HP_DISTS), string)
    if not len(dist_type):
        _LOGGER.exception("unsupported parameter distribution in hyperopt " + \
                         f"string:\n{string}\nSupported distributions are " + \
                         f"{', '.join(SUPPORTED_HP_DISTS)}\n")       
        raise ValueError
    
    # Handle categorical distributions for choice and pchoice
    if 'switch' in dist_type:
        if 'categorical' in dist_type:
            return 'pchoice', []
        else:
            return 'choice', []

    # Get all the Literal{} entries
    string = ' '.join(re.findall("Literal{-*\d+\.*\d*}", string))

    # Parse all the numbers within the {} and map to float
    parsed = list(map(float, re.findall("[Literal{}](-*\d+\.*\d*)", string)))
    
    return dist_type[0], parsed