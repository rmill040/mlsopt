from hyperopt import hp

# Dictionary to map distribution names to hyperopt functions
HP_DISTS = {
    'loguniform'  : hp.loguniform,
    'qloguniform' : hp.qloguniform,
    'uniform'     : hp.uniform,
    'choice'      : hp.choice,
    'pchoice'     : hp.pchoice,
    'quniform'    : hp.quniform
}

# Supported hyperopt distributions for analysis
SUPPORTED_HP_DISTS = [
    "switch",
    "categorical",
    "quniform",
    "uniform",
    "loguniform",
    "qloguniform"
]