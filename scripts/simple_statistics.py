import scipy.stats as stats
import numpy as np
from scipy.stats import shapiro
from matplotlib import pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns

class StatsAnalysis:

    @staticmethod
    def get_scatter_plot(df_data, factor_labels, independent_labels):

        if not isinstance(df_data, pd.DataFrame):
            try:
                pd.DataFrame(data=df_data)
            except:
                print("Unable to catch data")

        palette = ['#e41a1c', '#377eb8', '#4eae4b', '#994fa1', '#ff8101', '#fdfc33', '#a8572c', '#f482be', '#999999']

        # for label in factor_labels:
        #     sns.pairplot(df_data, x_vars=label, y_vars=independent_labels)
        #     plt.show()

        if len(independent_labels)>8:
            n_factors = len(independent_labels)//2
            df_data.boxplot(column=list(independent_labels[:n_factors]))
            plt.show()
            df_data.boxplot(column=list(independent_labels[n_factors:]))
            plt.show()

        for i, group in enumerate(df_data.groupby('group')):
            plt.title('_data of {}'.format(group[0]))

