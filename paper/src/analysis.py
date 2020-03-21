import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from xgboost import XGBClassifier

# Package imports 
from mlsopt.samplers import PipelineSampler, XGBClassifierSampler
from mlsopt.optimizers import RSOptimizer
from mlsopt.utils import STATUS_OK

# Constants
SEED     = 1718
MAIN_DIR = Path(__file__).resolve().parents[1]
SAVE_DIR = join(MAIN_DIR, 'results')


def rs_vs_rsdu():
    """Create plot of random search vs random search with updates.
    """
    # Load toy data
    X, y = load_breast_cancer(return_X_y=True)

    # Define samplers
    hp_sampler = XGBClassifierSampler(dynamic_update=True, seed=SEED)
    sampler    = PipelineSampler(seed=SEED)\
                    .register_sampler(hp_sampler, name='hp')

    def fn(space):
        """Function to optimize.
        """
        clf    = XGBClassifier(**space['hp'])
        metric = cross_val_score(clf, X, y, cv=5, n_jobs=5).mean()
        return {
            'status'  : STATUS_OK,
            'metric'  : metric,
            'message' : ''
        }

    # Random search without updates
    n_configurations = 10
    max_iterations   = 10
    opt = RSOptimizer(n_configurations=n_configurations, 
                      max_iterations=max_iterations, 
                      dynamic_update=False, 
                      n_jobs=-1,
                      verbose=1,
                      seed=SEED)
    opt.search(objective=fn, sampler=sampler, lower_is_better=False)
    df1 = opt.get_df()
    del opt

    # Random search with updates
    opt = RSOptimizer(n_configurations=n_configurations, 
                      max_iterations=max_iterations, 
                      dynamic_update=True, 
                      n_jobs=-1,
                      verbose=1,
                      seed=SEED)
    opt.search(objective=fn, sampler=sampler, lower_is_better=False)
    df2 = opt.get_df()

    # Boxplot and swarmplot
    sns.set(font_scale=1.5)
    f, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(19.20, 9.83))    
    mu1   = df1.groupby('iteration')['metric'].mean()
    sns.boxplot(x='iteration', y='metric', data=df1, ax=ax[0, 0])
    sns.swarmplot(x='iteration', y='metric', data=df1, color=".25", ax=ax[0, 0])
    ax[0, 1].plot(range(max_iterations), mu1)

    mu2 = df2.groupby('iteration')['metric'].mean()
    sns.boxplot(x='iteration', y='metric', data=df2, ax=ax[1, 0])
    sns.swarmplot(x='iteration', y='metric', data=df2, color=".25", ax=ax[1, 0])
    ax[1, 1].plot(range(max_iterations), mu2)

    # Decorate plots
    ax[0, 0].set(xlabel="Iteration", ylabel="Accuracy", title="Dynamic Updating=False")
    ax[0, 1].set(xlabel="Iteration", title="Mean Accuracy by Iteration")
    ax[1, 0].set(xlabel="Iteration", ylabel="Accuracy", title="Dynamic Updating=True")
    ax[1, 1].set(xlabel="Iteration", title="Mean Accuracy by Iteration")
    plt.tight_layout()
    plt.show()

    # Make plot
    plt.savefig(join(SAVE_DIR, 'rs_vs_rsdu.png'), dpi=200)


def main():
    """Creates figures and tables for MODSIM World 2020 paper.
    """

    df = pd.read_csv(join(SAVE_DIR, 'experiment1.csv'))

    # Create adjusted time for fs_with_gs method
    i                 = df['method'] == 'fs_with_gs'
    df.loc[i, 'time'] = df.loc[i, 'time'] * (200 / 216)
    
    # Table -- Calculate metric by method and data set
    print("*"*30)
    print("Table: Metric by Method and Data Set")
    print("*"*30)
    df_pivot = pd.pivot_table(data=df, columns='method', index='name', values='metric')
    df_pivot = df_pivot[['fs_with_gs', 'tpe', 'rs', 'rsdu', 'ga', 'pso']]
    df_pivot = df_pivot.reindex(index=['ALLAML.csv', 
                                       'cancer.csv', 
                                       'CLL_SUB_111.csv', 
                                       'madelon.csv', 
                                       'orlraws10P.csv', 
                                       'pixraw10P.csv',
                                       'spam.csv', 
                                       'TOX_171.csv', 
                                       'warpAR10P.csv', 
                                       'warpPIE10P.csv', 
                                       'Yale.csv', 
                                       'sim_n100_p200_i2.csv', 
                                       'sim_n100_p200_i20.csv', 
                                       'sim_n100_p500_i5.csv', 
                                       'sim_n100_p500_i50.csv'])
    print(df_pivot.round(4))
    print("\n")
    
    # Table -- Calculate time by method and data set
    print("*"*30)
    print("Table: Time by Method and Data Set")
    print("*"*30)
    df_pivot = pd.pivot_table(data=df, columns='method', index='name', values='time')
    df_pivot = df_pivot[['fs_with_gs', 'tpe', 'rs', 'rsdu', 'ga', 'pso']]
    df_pivot = df_pivot.reindex(index=['ALLAML.csv', 
                                       'cancer.csv', 
                                       'CLL_SUB_111.csv', 
                                       'madelon.csv', 
                                       'orlraws10P.csv', 
                                       'pixraw10P.csv',
                                       'spam.csv', 
                                       'TOX_171.csv', 
                                       'warpAR10P.csv', 
                                       'warpPIE10P.csv', 
                                       'Yale.csv', 
                                       'sim_n100_p200_i2.csv', 
                                       'sim_n100_p200_i20.csv', 
                                       'sim_n100_p500_i5.csv', 
                                       'sim_n100_p500_i50.csv'])
    print(df_pivot.round(2))
    print("\n")

    # Table -- Calculate number of winning methods across data sets
    print("*"*30)
    print("Table: TIME - Number of 'Winners' Across Data Sets")
    print("*"*30)
    winners = {method: 0 for method in df['method'].unique()}
    for name in df['name'].unique():
        tmp = df[df['name'] == name].sort_values(by='time')
        print(tmp[['name', 'method', 'metric']].round(4))
        names = tmp[tmp['time'] == tmp['time'].min()]['method'].tolist()
        for name in names:
            winners[name] += 1
    print(winners)
    print("\n")
    
    # Table -- Calculate number of winning methods across data sets
    print("*"*30)
    print("Table: METRIC - Number of 'Winners' Across Data Sets")
    print("*"*30)
    winners = {method: 0 for method in df['method'].unique()}
    for name in df['name'].unique():
        tmp = df[df['name'] == name].sort_values(by='metric')
        print(tmp[['name', 'method', 'metric']].round(4))
        names = tmp[tmp['metric'] == tmp['metric'].max()]['method'].tolist()
        for name in names:
            winners[name] += 1
    print(winners)
    print("\n")

    # Table -- Calculate timings by method across data sets
    print("*"*30)
    print("Table: Timings by Method")
    print("*"*30)

    print(df.groupby('method')['time'].describe().reset_index().round(4))
    print("*"*30)

    # Table -- Calculate metrics by method across data sets
    print("*"*30)
    print("Table: Metrics by Method")
    print("*"*30)

    print(df.groupby('method')['metric'].describe().reset_index().round(4))
    print("*"*30)

    # Table -- Index of best result
    df_sub              = df[~df['method'].isin(['fs_with_gs'])]
    df_sub['idx_ratio'] = df_sub['idx'] / 200
    print("*"*30)
    print("Table: Round Best Metric by Method")
    print("*"*30)

    print(df_sub.groupby('method')['idx_ratio'].describe().reset_index().round(4))
    print("*"*30)
    
    # Table -- Calculate time by method and data set
    print("*"*30)
    print("Table: Percent Best Iteration by Method and Data Set")
    print("*"*30)
    df_pivot = pd.pivot_table(data=df_sub, columns='method', index='name', values='idx_ratio')
    df_pivot = df_pivot[['tpe', 'rs', 'rsdu', 'ga', 'pso']]
    df_pivot = df_pivot.reindex(index=['ALLAML.csv', 
                                       'cancer.csv', 
                                       'CLL_SUB_111.csv', 
                                       'madelon.csv', 
                                       'orlraws10P.csv', 
                                       'pixraw10P.csv',
                                       'spam.csv', 
                                       'TOX_171.csv', 
                                       'warpAR10P.csv', 
                                       'warpPIE10P.csv', 
                                       'Yale.csv', 
                                       'sim_n100_p200_i2.csv', 
                                       'sim_n100_p200_i20.csv', 
                                       'sim_n100_p500_i5.csv', 
                                       'sim_n100_p500_i50.csv'])
    print(100 * df_pivot.round(2))
    print("\n")

    # Table -- Calculate number of winning methods across data sets
    print("*"*30)
    print("Table: PERCENT BEST ITERATION - Number of 'Winners' Across Data Sets")
    print("*"*30)
    winners = {method: 0 for method in df_sub['method'].unique()}
    for name in df_sub['name'].unique():
        tmp = df_sub[df_sub['name'] == name].sort_values(by='idx_ratio')
        print(tmp[['name', 'method', 'metric']].round(4))
        names = tmp[tmp['idx_ratio'] == tmp['idx_ratio'].max()]['method'].tolist()
        for name in names:
            winners[name] += 1
    print(winners)
    print("\n")

    # Figure -- Random search with and without dynamic updates
    rs_vs_rsdu()

    # ANOVA on metrics
    print("ANOVA: metric ~ C(method)")
    model = ols('metric ~ C(method)', data=df).fit()
    print(model.summary())

    # ANOVA on time
    print("ANOVA: time ~ C(method)")
    model = ols('time ~ C(method)', data=df).fit()
    print(model.summary())

    print("Tukey: time ~ C(method)")
    mc     = MultiComparison(df['time'], df['method'])
    result = mc.tukeyhsd()
    print(result)

    # ANOVA percent best iteration
    print("ANOVA: idx_ratio ~ C(method)")
    model  = ols('idx_ratio ~ C(method)', data=df_sub).fit()
    print(model.summary())

    print("Tukey: idx_ratio ~ C(method)")
    mc     = MultiComparison(df_sub['idx_ratio'], df_sub['method'])
    result = mc.tukeyhsd()
    print(result)


if __name__ == "__main__":
    main()