from scripts.data_import import plvDataImport
from scripts.consts import plvTresholdAnova2Data
import os as os
import numpy as np

def main():
    dirpath = os.getcwd()
    filename = "PLV_threshold_ANOVA2.xlsx"
    #sheetname = 'PLV_threshold_ANOVA2'
    factor_idx = plvTresholdAnova2Data.FACTOR_IDX
    anova2_data_import = plvDataImport()
    all_data = anova2_data_import.import_data_from_xslx(filename, dirpath)
    factor_labels, independent_labels = anova2_data_import.get_labels(factor_idx)


if __name__ == "__main__":
    main()
