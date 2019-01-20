from openpyxl import load_workbook
import pandas as pd
import numpy as np
from consts import indiciesList
# import matplotlib.pyplot as plt
# import scipy
# import plotly
# #from stats import probplot
# from pandas.tools import plotting
# import matplotlib
# import plotly.plotly as py
# #plotly.tools.set_credentials_file(username='uzuchowicz', api_key='mkt4BEiLw1kbLYLP5BEf')
# from plotly.tools import FigureFactory as FF
# import scipy.stats as sstats
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.ticker as mticker
# import scipy
# import pylab
# import functions as fct
# import plotting as plot
import os as os
import sys as sys
import logging

class plvDataImport:
    def __init__(self):
        self.filename = ""
        self.datapath = "/data/"
        self.dirpath = ""
        self.indices = []
        self.factor_labels = []
        self.independent_labels = []
        self.data = []

    def import_filepath(self, **kwargs):
        if not self.dirpath:
            self.dirpath = kwargs.get('dirpath')
            print("Saving path...")
        if not self.filename:
            self.filename = kwargs.get('filename')
            print("Saving filename...")
        filepath = self.dirpath + self.datapath + self.filename

        return filepath


    def import_data_from_xslx(self, filename=None, dirpath=None):

        filepath = self.import_filepath(filename=filename, dirpath=dirpath)
        try:
            excel_data = pd.read_excel(os.path.join(filepath))
            # excel_data = pd.read_excel(os.path.join(filepath), skiprows=0)
            # self.data = np.matrix(excel_data)
            self.data = excel_data
            print("Importing data of size {0}".format(np.shape(self.data)))

        except IOError:
            print("Error: can\'t find file or read data")
            sys.exit(1)
        except Exception as e:
            print("Unable to open file...")
            logging.exception(e)
            sys.exit(1)

        return excel_data

    def get_labels(self, factor_idx, filename = None, dirpath=None):
        filepath = self.import_filepath(filename=filename, dirpath=dirpath)
        try:
            labels = pd.read_excel(os.path.join(filepath), header=0)
        except IOError:
            print("Error: can\'t find file or read data")
            sys.exit(1)
        except Exception as e:
            print("Unable to open file...")
            logging.exception(e)
            sys.exit(1)
        labels = tuple(labels)
        factor_labels = labels[:factor_idx]
        independent_labels = labels[factor_idx:]
        self.factor_labels = factor_labels
        self.independent_labels = independent_labels
        print("Data factors: {0}".format(factor_labels))
        print("Data contains indices: {0}".format(independent_labels))

        return factor_labels, independent_labels
