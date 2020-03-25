import numpy as np
from typing import Union

# Package imports
from .constants import T_NUMERIC


def pandas_safe_iloc(df, iloc):
    """ADD HERE
    
    Parameters
    ----------
    
    Returns
    -------
    """        
    return df.astype(object).iloc[iloc]


def pandas_cast_dtypes(df, dtypes):
    """ADD HERE
    
    Parameters
    ----------
    
    Returns
    -------
    """
    df_ = df.copy()
    for column in df.columns:
        df_[column] = df[column].astype(dtypes[column])
    return df_


def sigmoid(x: Union[np.ndarray, T_NUMERIC]) -> Union[np.ndarray, T_NUMERIC]:
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    return 1 / (1 + np.exp(-x))