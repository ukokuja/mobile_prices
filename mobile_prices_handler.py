import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

CATEGORICAL_KEYS = ['bluetooth', 'gen', 'cores', 'speed', 'sim', 'screen']
ORDINAL_SORTS = {
    'gen': [2, 3, 4],
    'cores': ['single','dual', 'triple','quad','penta', 'hexa', 'hepta','octa'],
    'speed': ['low', 'medium', 'high'],
    'wifi': ['none', 'b', 'a','g', 'n']
}


class MobilePricesHandler:
    """
    General mobile prices handler
    """
    def __init__(self):
        """
        Reads the file mobile_price_1.csv, cleans and transform it.
        """
        self._df = pd.read_csv("mobile_price_1.csv", sep=",", index_col="id")
        self._pre_process()

    def _pre_process(self):
        """
        Do some modifications on the dataframe
        :return:
        """
        self._clean_data()
        self._transform()

    def _transform(self):
        """
        * Transform data for better analysis
        """
        self._df.price = self._df.price.astype(int)  # lots of decimals
        self._df['f_camera'] = self._df['f_camera'].fillna(0)
        self._df['camera'] = self._df['camera'].fillna(0)

    def _clean_data(self):
        """
        * Cleans some rows that can be considered as noise.
        :return:
        """

        # probably noise, there's no phone with < 100 px
        self._df = self._df.loc[(self._df.px_height > 100)]

        # probably noise, there's no phone with < 2cm screen width
        self._df = self._df.loc[(self._df.sc_w > 2)]

    def _get_ordered_correlative_by_key(self, key='price', **kwargs):
        """
        :param kwargs: Filter params
        :param key: key to filter
        :return: Returns correlation of one feature compared with all the other relevant ones, sorted.
        """
        corr = self.get_correlation_df(**kwargs)
        return corr[key].sort_values(ascending=False)

    def get_correlation_df(self, **kwargs):
        """
        :param kwargs: Filter params
        :return: Returns correlation dataframe
        """
        df = self._df.copy()
        if kwargs.get('ignore_columns'):
            df = df[df.columns.difference(kwargs.get('ignore_columns'))]
        if kwargs.get('only_numerical'):
            df = df.select_dtypes(np.number)
        return df.corr().abs()

    def show_correlation_heatmap(self, title='', **kwargs):
        """
        Shows correlation graph using filters
        :param title: Graph filter
        :param kwargs: Filter params
        :return:
        """
        corr = self.get_correlation_df(**kwargs)
        corr_list = self._get_ordered_correlative_by_key(**kwargs)
        mask = np.triu(corr)
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmax=corr_list[1],
                    square=True, linewidths=.5).set_title(title)
        plt.show()

    def map_ordinal_categorical(self, sorts, postfix=''):
        """
        Maps ordinal categorical features
        :param sorts: Order of keys in categorical features
        :param postfix: postfix to add after each feature key
        """
        for key, value in sorts.items():
            self._df['{}_{}'.format(key, postfix)] = pd.Categorical(self._df[key],
                                                                    ordered=True,
                                                                    categories=value
                                                                    ).codes

    def map_nominal_categorical(self):
        """
        Maps nominal categorical features (binary and one-hot)
        :return:
        """
        self._df['bluetooth_bin'] = self._df['bluetooth'] == "Yes"
        self._df = pd.get_dummies(self._df, columns=['screen'], prefix=['screen'])
        self._df = pd.get_dummies(self._df, columns=['sim'], prefix=['sim'])

    def map_categorical_columns(self):
        """
        Map every categorical feature and drop the original ones
        :return:
        """
        self.map_ordinal_categorical(postfix='ord', sorts=ORDINAL_SORTS)
        self.map_nominal_categorical()
        for col in CATEGORICAL_KEYS:
            if col in self._df.columns:
                self._df.drop(columns=col, inplace=True)

    def show_relation_by_keys(self, title, x='speed', y='price'):
        """
        Plot the relation of two features
        :param title: Graph title
        :param x: First feature
        :param y: Second feature
        :return:
        """
        sns.jointplot(x=x, y=y, data=self._df)
        plt.subplots_adjust(top=0.9)
        plt.gcf().suptitle(title)  # can also get the figure from plt.gcf()
        plt.show()
