from mlsopt.sampler import FeatureSampler, XGBClassifierSampler


def main():
    # sampler = FeatureSampler(n_features=10)
    sampler = XGBClassifierSampler(dynamic_update=True)
    sampler.update_space()


if __name__ == "__main__":
    main()