from scripts.data_import import plvDataImport
from scripts.consts import plvTresholdAnova2Data, indiciesList
import os as os
from distribution_analysis import distributionAnalysis
from variance_analysis import varianceAnalysis
import numpy as np

def main():
    dirpath = os.getcwd()
    filename = "PLV_threshold_ANOVA2.xlsx"
    #sheetname = 'PLV_threshold_ANOVA2'
    factor_idx = plvTresholdAnova2Data.FACTOR_IDX

    anova2_data_import = plvDataImport()
    all_data = anova2_data_import.import_data_from_xslx(filename, dirpath)
    factor_labels, independent_labels = anova2_data_import.get_labels(factor_idx)
    param_name = indiciesList.DLR.value
    param_idx = factor_idx
    print(param_name)
    if param_name in independent_labels:
        param_idx = + independent_labels.index(param_name)
    #all_data = all_data[all_data["group"] == 3]

    param_data = all_data[param_name].values
    #data_response_MDD =param_data[param_data["group"] == 1]
    is_normal = distributionAnalysis().test_normality(param_data)
    varianceAnalysis().variance_test(is_normal, all_data, param_name, "band", "group", "condition")




if __name__ == "__main__":
    main()
