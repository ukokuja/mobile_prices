import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ex_3.mobile_prices_handler import MobilePricesHandler



class Task_4(MobilePricesHandler):

    def _initial_mapping(self):
        self.map_categorical_columns()
        self._df['price_log'] = np.log(self._df['price'])

    def show_4_dimensions_plot(self):
        self._initial_mapping()
        plt.scatter(x=self._df['px_width'], y=self._df['px_height'], s=(self._df['cores_ord'] + 1) * 100,
                    c=self._df['price_log'], alpha=0.3,
                    cmap='RdGy')
        plt.colorbar()
        plt.show()


    def check_transformation(self):
        transformed_df = pd.read_csv("mobile_price_2.csv", sep=",", index_col="id")
        self._df = pd.merge(self._df, transformed_df, on='id', how='left')
        self._df['price_rate'] = np.round(self._df['price_2'] / self._df['price'], 2)
        self.show_correlation_heatmap(only_numerical=False, ignore_columns=None)
        self.show_relation_by_keys(x='camera', y='price_rate')

if __name__ == '__main__':
    mbp = Task_4()
    mbp.show_4_dimensions_plot()
    mbp.check_transformation()

