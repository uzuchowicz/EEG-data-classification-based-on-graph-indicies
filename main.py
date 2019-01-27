from scripts.data_import import PlvDataImport
from scripts.consts import PlvThresholdAnova2Data, IndiciesList
import os as os
from distribution_analysis import DistributionAnalysis
from variance_analysis import VarianceAnalysis
import numpy as np
from knn_classifier import Classifier
#from spark.sql.function import *
from pyspark import *
from pyspark.sql.functions import *
from pyspark import SparkConf, SparkContext
from simple_statistics import StatsAnalysis
from spark_analysis import SparkAnalysis


def main():
    dir_path = os.getcwd()
    filename = "PLV_threshold_ANOVA2.xlsx"
    factor_idx = PlvThresholdAnova2Data.FACTOR_IDX
    table_name = "PLV_treshold"
    anova2_data_import = PlvDataImport()
    all_data = anova2_data_import.import_data_from_xslx(filename, dir_path)
    factor_labels, independent_labels = anova2_data_import.get_labels(factor_idx, filename, dir_path)

    csv_file_path = '/home/ulek/EEG-data-classification-based-on-graph-indicies/data/PLV_threshold_ANOVA2.csv'
    SparkAnalysis().import_data_from_csv(csv_file_path)
    # StatsAnalysis().get_scatter_plot(all_data, factor_labels, independent_labels)
    # param_name = IndiciesList.DLR.value
    # param_data = all_data[param_name].values
    # param_data = SparkAnalysis().import_data(param_data)
    # data_response_MDD =param_data[param_data["group"] == 1]
    # is_normal = distributionAnalysis().test_normality(param_data)
    # varianceAnalysis().variance_test(is_normal, all_data, param_name, "band", "group", "condition")


if __name__ == "__main__":
    main()
