import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ex_3.main import MobilePricesHandler

categorical = ['bluetooth', 'gen', 'cores', 'speed', 'sim', 'screen']


class Task_2(MobilePricesHandler):

    def _map_ordinal_categorical(self):
        self._df['bluetooth'].replace({'No': 0, 'Yes': 1}, inplace=True)
        self._df['cores'].replace(
            {'single': 1, 'dual': 2, 'triple': 3, 'quad': 4, 'penta': 5, 'hexa': 6, 'hepta': 7, 'octa': 8},
            inplace=True)
        self._df['speed'].replace({'low': 1, 'medium': 2, 'high': 3}, inplace=True)
        self._df['sim'].replace({'Single': 1, 'Dual': 2}, inplace=True)
        self._df['wifi'].replace({'none': 0, 'b': 1, 'a': 2, 'g': 3, 'n': 4}, inplace=True)

    def show_correlation_heatmap_including_ordinal_categorical(self):
        self._map_ordinal_categorical()
        self.show_correlation_heatmap()

    def show_most_correlated_features(self, start, key):
        corr_list = self._get_ordered_correlative_by_key(key=key)
        print(corr_list[corr_list > start])

    def show_relation_by_keys(self, x='speed', y='price'):
        sns.jointplot(x=x, y=y, data=self._df)
        plt.show()

    def show_violin_plot_for_key(self, key, order):
        sns.violinplot(x=key, y='price',
                       order=order,
                       data=self._df)
        plt.show()

    def show_pivot_table(self):
        index_list = ['ram', 'battery_power']
        df = self._df
        for index in index_list:
            df[index] = pd.qcut(df[index], 5)
        print(np.round(pd.pivot_table(df, values='price', index=index_list,
                                      columns='gen', aggfunc=np.mean), 1))


if __name__ == '__main__':
    mbp = Task_2()
    # 1.	Plot a correlation heatmap of the data set and include it.
    mbp.show_correlation_heatmap(ignore_cols=categorical)
    # 2.	Which features would you say are correlated with the device price?
    mbp.show_most_correlated_features(start=0.3, key='price')
    # 3.	Are there features not shown in the correlation matrix
    # that are correlated with the price? If so, what are they?
    mbp.show_correlation_heatmap_including_ordinal_categorical()
    # 4.	For each feature correlated with the price, plot its relationship with price.
    # RAM
    mbp.show_relation_by_keys('ram')
    # GEN
    mbp.show_violin_plot_for_key('gen', [2, 3, 4])
    # Battery Power
    mbp.show_relation_by_keys('battery_power')
    # 5.	Select 3 features that are correlated with price
    # and create a pivot table showing average pricewith relation
    # to cross sections of those 3 features (remember to divide numerical
    # features into cuts, for example quartile cuts).
    mbp.show_pivot_table()
