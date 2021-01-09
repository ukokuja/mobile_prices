import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ex_3.mobile_prices_handler import MobilePricesHandler, CATEGORICAL


class Task_2(MobilePricesHandler):


    def show_correlation_heatmap_including_ordinal_categorical(self):
        self.map_categorical_columns()
        self.show_correlation_heatmap()

    def show_most_correlated_features(self, start, key):
        corr_list = self._get_ordered_correlative_by_key(key=key)
        print(corr_list[corr_list > start])

    def show_violin_plot_for_key(self, key, order):
        sns.violinplot(x=key, y='price',
                       order=order,
                       data=self._df)
        plt.show()

    def show_pivot_table(self):
        index_list = ['ram', 'battery_power']
        df = self._df.copy()
        for index in index_list:
            df[index] = pd.qcut(df[index], 5)
        pivot_table = np.round(pd.pivot_table(df, values='price', index=index_list,
                                      columns='gen', aggfunc=np.mean), 1)
        print(pivot_table)
        pivot_table.to_csv('task_2.5_pivot_table.csv')


if __name__ == '__main__':
    mbp = Task_2()
    # 1.	Plot a correlation heatmap of the data set and include it.
    mbp.show_correlation_heatmap(ignore_cols=CATEGORICAL, only_numerical=True)
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
