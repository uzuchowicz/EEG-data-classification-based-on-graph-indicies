import scipy.stats as stats
import numpy as np
from scipy.stats import shapiro
from matplotlib import pyplot as plt
import statsmodels.api as sm


class DistributionAnalysis:

    def test_normality(self, data, use_dagostino=False, modify_to_normality=False):
        alpha = 0.05
        is_normal = False

        if use_dagostino:
            stat, p = stats.mstats.normaltest(data)
            print("Using D'Agostino and Pearson test for verifying normality...")
        elif len(data) < 5000:
            stat, p = shapiro(data)
            print("Using Shapiro-Wilk test for verifying normality...")

        else:
            stat, p = sm.stats.lilliefors(data)
            print("Using Lilliefors test for verifying normality...")

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        if p > alpha:
            is_normal = True
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

        if modify_to_normality:
            is_normal = self.check_modified_data(data)

        plt.hist(data)
        plt.show()
        return is_normal

    def check_modified_data(self, data):
        data = np.exp(data)
        # power transform
        data = stats.boxcox(data, 0)
        modify_to_normality = False

        assert not modify_to_normality, "Could not modify data many times!"
        is_normal = self.test_normality(data)
        return is_normal
