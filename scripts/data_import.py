from openpyxl import load_workbook
import pandas as pd
import numpy as np
from consts import IndiciesList
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
import sqlite3


class PlvDataImport:
    def __init__(self):
        self.filename = ""
        self.data_path = "/data/"
        self.dir_path = ""
        self.indices = []
        self.factor_labels = []
        self.independent_labels = []
        self.data = []

    def get_file_path(self):
        return self.dir_path + self.data_path + self.filename

    def import_file_path(self, **kwargs):
        if not self.dir_path:
            self.dir_path = kwargs.get('dir_path')
            print("Saving path...")
        if not self.filename:
            self.filename = kwargs.get('filename')
            print("Saving filename...")
        file_path = self.dir_path + self.data_path + self.filename

        return file_path

    def import_data_from_xslx(self, filename=None, dir_path=None):

        file_path = self.import_file_path(filename=filename, dir_path=dir_path)
        try:
            excel_data = pd.read_excel(os.path.join(file_path))
            # excel_data = pd.read_excel(os.path.join(file_path), skiprows=0)
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

    def get_labels(self, factor_idx, filename=None, dir_path=None):
        file_path = self.import_file_path(filename=filename, dir_path=dir_path)
        try:
            labels = pd.read_excel(os.path.join(file_path), header=0)
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

    def import_data_from_db(self, table_name, filename=None, dir_path=None):

        file_path = self.import_file_path(filename=filename, dir_path=dir_path)
        try:
            connection = sqlite3.connect(file_path)
            connection.text_factory = sqlite3.OptimizedUnicode
            print(file_path)
            db_data = pd.read_sql_query('SELECT * from '+table_name, connection)

            self.data = db_data
            print("Importing data of size {0}".format(np.shape(self.data)))

        except IOError:
            print("Error: can\'t find file or read data")
            sys.exit(1)

        except Exception as e:
            print("Unable to open file...")
            logging.exception(e)
            sys.exit(1)

        return db_data

    def get_labels_from_db(self,  table_name, factor_idx, filename=None, dir_path=None):
        file_path = self.import_file_path(filename=filename, dir_path=dir_path)
        try:
            connection = sqlite3.connect(file_path)
            cursor = connection.cursor()
            cursor.execute('SELECT * from '+table_name)
            labels = list(map(lambda x: x[0], cursor.description))
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
