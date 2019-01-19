

class varianceAnalysis():
    def __init__(self):
        self.is_normal = False

    def variance_test(self, is_normal, data):

        if is_normal:
            varianceAnalysis.anova_test(data)
        else:
            varianceAnalysis.wk_anova_test(data)
        return True


    @staticmethod
    def anova_test(data):
        pass

    @staticmethod
    def wk_anova_test(data):
        pass
