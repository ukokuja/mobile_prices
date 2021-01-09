import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

CATEGORICAL = ['bluetooth', 'gen', 'cores', 'speed', 'sim', 'screen']
SORTS = {
    'gen': [2, 3, 4],
    'cores': ['single','dual', 'triple','quad','penta', 'hexa', 'hepta','octa'],
    'speed': ['low', 'medium', 'high'],
    'wifi': ['none', 'b', 'a','g', 'n']
}


class MobilePricesHandler:

    def __init__(self):
        self._df = pd.read_csv("mobile_price_1.csv", sep=",", index_col="id")
        self._pre_process()

    def _pre_process(self):
        self._clean_data()
        self._transform()

    def _transform(self):
        self._df.price = self._df.price.astype(int)  # lots of decimals

    def _clean_data(self):
        self._df['f_camera'] = self._df['f_camera'].fillna(0)
        self._df['camera'] = self._df['camera'].fillna(0)

        # probably noise, there's no phone with < 100 px
        self._df = self._df.loc[(self._df.px_height > 100)]

        # probably noise, there's no phone with < 2cm screen width
        self._df = self._df.loc[(self._df.sc_w > 2)]

    def _get_ordered_correlative_by_key(self, only_numerical, ignore_columns, key='price'):
        corr = self.get_correlation_df(only_numerical=only_numerical, ignore_columns=ignore_columns)
        return corr[key].sort_values(ascending=False)

    def get_correlation_df(self, ignore_columns=None, only_numerical=False):
        df = self._df.copy()
        if ignore_columns:
            df = df[df.columns.difference(ignore_columns)]
        if only_numerical:
            df = df.select_dtypes(np.number)
        return df.corr().abs()

    def show_correlation_heatmap(self, ignore_columns, only_numerical):
        corr = self.get_correlation_df(only_numerical=only_numerical, ignore_columns=ignore_columns)
        corr_list = self._get_ordered_correlative_by_key(only_numerical=only_numerical, ignore_columns=ignore_columns)
        mask = np.triu(corr)
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmax=corr_list[1],
                    square=True, linewidths=.5)
        plt.show()

    def map_ordinal_categorical(self, sorts, postfix=''):
        for key, value in sorts.items():
            self._df['{}_{}'.format(key, postfix)] = pd.Categorical(self._df[key],
                                                                    ordered=True,
                                                                    categories=value
                                                                    ).codes

    def map_nominal_categorical(self):
        self._df['bluetooth_bin'] = self._df['bluetooth'] == "Yes"
        self._df = pd.get_dummies(self._df, columns=['screen'], prefix=['screen'])
        self._df = pd.get_dummies(self._df, columns=['sim'], prefix=['sim'])

    def map_categorical_columns(self):
        self.map_ordinal_categorical(postfix='ord', sorts=SORTS)
        self.map_nominal_categorical()
        for col in CATEGORICAL:
            if col in self._df.columns:
                self._df.drop(columns=col, inplace=True)

    def show_relation_by_keys(self, x='speed', y='price'):
        sns.jointplot(x=x, y=y, data=self._df)
        plt.show()