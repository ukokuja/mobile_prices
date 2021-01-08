import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MobilePricesHandler:
    def __init__(self):
        self._df = pd.read_csv("mobile_price_1.csv", sep=",", index_col="id")
        self._pre_process()

    def _pre_process(self):
        self._clean_data()
        self._transform()

    def _transform(self):
        self._df.price = self._df.price.round(0)  # lots of decimals
        self._df.ram = self._df.ram.round(0)  # lots of decimals
        self._df.memory = np.log(self._df.memory)  # big difference between min and max

    def _clean_data(self):
        self._df['f_camera'] = self._df['f_camera'].fillna(0)
        self._df['camera'] = self._df['camera'].fillna(0)
        self._df = self._df.loc[(self._df.px_height > 10)]

    def _get_ordered_correlative_by_key(self, key='price'):
        corr = self.get_correlation_df()
        return corr[key].sort_values(ascending=False)

    def get_correlation_df(self, only_numerical=True):
        df = self._df
        if only_numerical:
            df = df.select_dtypes(np.number)
        return df.corr().abs()


    def show_correlation_heatmap(self, ignore_cols=[], only_numerical=True):
        corr = self.get_correlation_df(only_numerical=only_numerical)
        corr = corr[corr.columns.difference(ignore_cols)]
        corr_list = self._get_ordered_correlative_by_key()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmax=corr_list[1],
                    square=True, linewidths=.5)
        plt.show()