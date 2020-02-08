import pandas as pd

# Package imports
from mlsopt.sampler import (
    FeatureSampler, XGBClassifierSampler
)


def main():

    sampler = XGBClassifierSampler(dynamic_update=True)
    data    = pd.DataFrame([sampler.sample_space() for _ in range(50)])
    import matplotlib.pyplot as plt
    
    print(data['n_estimators'].describe())
    plt.hist(data['n_estimators']); plt.show()


if __name__ == "__main__":
    main()