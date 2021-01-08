import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ex_3.main import MobilePricesHandler

categorical = ['bluetooth', 'gen', 'cores', 'speed', 'sim', 'screen']


class Task_3(MobilePricesHandler):

    def map_ordinal_categorical(self):
        self._df['gen_ord'] = self._df['gen']
        self._df['cores_ord']= self._df['cores'].replace(
            {'single': 1, 'dual': 2, 'triple': 3, 'quad': 4, 'penta': 5, 'hexa': 6, 'hepta': 7, 'octa': 8})
        self._df['speed_ord'] = self._df['speed'].replace({'low': 1, 'medium': 2, 'high': 3})
        self._df['wifi_ord'] = self._df['wifi'].replace({'none': 0, 'b': 1, 'a': 2, 'g': 3, 'n': 4})

    def map_nominal_categorical(self):
        self._df['bluetooth_bin'] = self._df['bluetooth'] == "Yes"
        self._df = pd.get_dummies(self._df, columns=['screen'], prefix=['screen'])
        self._df = pd.get_dummies(self._df, columns=['sim'], prefix=['sim'])

    def map_categorical_columns(self):
        self.map_ordinal_categorical()
        self.map_nominal_categorical()
        for col in categorical:
            if col in self._df.columns:
                self._df.drop(columns=col, inplace=True)


    def output_csv(self):
        self._df.to_csv("mobile_prices_converted.csv")

if __name__ == '__main__':
    mbp = Task_3()
    # 1. For each ordinal feature <O>, add a column to the
    # dataframe which holds the ordered values representing
    # each original value of F. This new column will be named
    # <O>_ord. (without the triangle brackets)
    #2. For each nominal feature <N>, add a binary column
    # OR one-hot encoding (whichever is relevant for that feature)
    # to the dataframe representing the original values.
    # Name binary columns <N>_bin, and prefix one-hot
    # encodings with <N>. (without the triangle brackets)
    mbp.map_categorical_columns()
    #3. Plot a correlation heatmap of the modified data set and include it.
    mbp.show_correlation_heatmap(only_numerical=False)
    #4. Save the entire dataframe to a csv file named “mobile_prices_converted.csv” and include it in the submission.
    # Make sure you don’t add a redundant index column.
    mbp.output_csv()

