import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import spm1d as spm

class varianceAnalysis():
    def __init__(self):
        self.is_normal = False

    def variance_test(self, is_normal, data, param, factor, factor2, factor3):

        if is_normal:
            varianceAnalysis.anova_test(data, param, factor, factor2, factor3)
        else:
            varianceAnalysis.wk_anova_test(data, param, factor, factor2, factor3)
        return True


    @staticmethod
    def anova_test(data, param, factor, factor2, factor3):
        varianceAnalysis.three_factors_anova(data, param, factor, factor2, factor3)
        return True

    @staticmethod
    def wk_anova_test(data, param, factor, factor2, factor3):
        varianceAnalysis.three_factors_kw_anova(data, param, factor, factor2, factor3)
        print("")
        return True


    @staticmethod
    def three_factors_anova(index_data, param, factor1, factor2, factor3):
        print("Computing ANOVA ... ")
        print('ANOVA:2 czynniki', factor1, '*', factor2,'*', factor3)
        print('_________________________________________________')

        for name_group in index_data.groupby(factor2):

            print(factor2, ':{}'.format(name_group[0]))
            group_index_data = index_data[index_data[factor2] == name_group[0]]
            #print('source_group0',source_group[0])

            for source_group in group_index_data.groupby(factor1):
                samples = [condition[1] for condition in source_group[1].groupby(factor3)[param]]

                f_val, p_val = stats.f_oneway(*samples)
                print(factor1, ': {} | F value : {:.3f} | p value : {:.3f}|'.format(source_group[0], f_val, p_val))

        sns.set(style="whitegrid")
        paper_rc = {'lines.linewidth': 0.5, 'lines.markersize': 15}
        sns.set_style("darkgrid")
        sns.set_context("paper", rc=paper_rc)
        sns.factorplot(x=factor1, y=param, hue=factor3, col=factor2, data=index_data, ci=95,capsize=.3, dodge=True)
        plt.grid(True, which="both", ls="-", c='w', color='w')
        plt.show()

    @staticmethod
    def three_factors_kw_anova(index_data, param, factor1, factor2, factor3):
        print("Computing KW ANOVA ... ")
        print('ANOVA:2 czynniki', factor1, '*', factor2, '*', factor3)
        print('_________________________________________________')

        for name_group in index_data.groupby(factor2):

            print(factor2, ':{}'.format(name_group[0]))
            group_index_data = index_data[index_data[factor2] == name_group[0]]
            # print('source_group0',source_group[0])

            for source_group in group_index_data.groupby(factor1):

                samples = [condition[1] for condition in source_group[1].groupby(factor3)[param]]
                f_val, p_val = stats.kruskal(*samples)
                print(factor1, ': {} | F value : {:.3f} | p value : {:.3f}|'.format(source_group[0], f_val, p_val))

        sns.set(style="whitegrid")
        paper_rc = {'lines.linewidth': 0.5, 'lines.markersize': 15}
        sns.set_style("darkgrid")
        sns.set_context("paper", rc=paper_rc)
        sns.factorplot(x=factor1, y=param, hue=factor3, col=factor2, data=index_data, ci=95, capsize=.3, dodge=True)

        plt.grid(True, which="both", ls="-", c='w', color='w')
        plt.show()
