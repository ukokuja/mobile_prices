import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mobile_prices_handler import MobilePricesHandler, CATEGORICAL_KEYS


class Task_2(MobilePricesHandler):


    def show_correlation_heatmap_including_ordinal_categorical(self):
        """
        Maps categorical columns, and shows a heatmap including them
        :return:
        """
        self.map_categorical_columns()
        self.show_correlation_heatmap(title='Heatmap including categorical features')

    def show_most_correlated_features(self, start, key):
        """
        Shows a list of most correlated features
        :param start: Minimum correlation to show
        :param key: Key to filter the most correlated features
        """
        corr_list = self._get_ordered_correlative_by_key(key=key)
        print(corr_list[corr_list > start])

    def show_violin_plot_for_key_vs_price(self, key, order, title, labels=[]):
        """
        Shows violin plot that compares the received key with the price
        :param key: The key to compare to price
        :param order: Order of violin elements
        :param title: Plot title
        :param labels: Labels of the violin elements
        :return:
        """
        plot = sns.violinplot(x=key, y='price',
                       order=order,
                       data=self._df)
        plot.set_title(title)
        if labels:
            plot.set_xticklabels(labels)
        plt.show()

    def show_pivot_table(self):
        """
        Shows and save pivot table using indexes ram and battery_power,
        Column gen_ord and values as price
        :return:
        """
        index_list = ['ram', 'battery_power']
        df = self._df.copy()
        for index in index_list:
            df[index] = pd.qcut(df[index], 5)
        pivot_table = np.round(pd.pivot_table(df, values='price', index=index_list,
                                      columns='gen_ord', aggfunc=np.mean), 1)
        print(pivot_table)
        pivot_table.to_csv('output_files/task_2.5_pivot_table.csv')


if __name__ == '__main__':
    mbp = Task_2()
    # 1.	Plot a correlation heatmap of the data set and include it.
    mbp.show_correlation_heatmap(title='Numerical heatmap', ignore_columns=CATEGORICAL_KEYS, only_numerical=True)
    # 2.	Which features would you say are correlated with the device price?
    mbp.show_most_correlated_features(start=0.3, key='price')
    # 3.	Are there features not shown in the correlation matrix
    # that are correlated with the price? If so, what are they?
    mbp.show_correlation_heatmap_including_ordinal_categorical()
    # 4.	For each feature correlated with the price, plot its relationship with price.
    # RAM
    mbp.show_relation_by_keys(x='ram', title='Ram vs Price')

    # GEN
    mbp.show_violin_plot_for_key_vs_price(key='gen_ord', order=[0, 1, 2], labels=[2, 3, 4], title='Gen violin plot')
    # Battery Power
    mbp.show_relation_by_keys(x='battery_power', title='Battery Power vs Price')
    # 5.	Select 3 features that are correlated with price
    # and create a pivot table showing average pricewith relation
    # to cross sections of those 3 features (remember to divide numerical
    # features into cuts, for example quartile cuts).
    mbp.show_pivot_table()
