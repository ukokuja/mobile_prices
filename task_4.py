import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mobile_prices_handler import MobilePricesHandler


class Task_4(MobilePricesHandler):

    def __init__(self):
        """
        Constructor of task 4
        """
        super().__init__()
        self.map_categorical_columns()

    def show_4_features_relation(self):
        """
        Show pair plot for features battery_power, ram, wifi and price
        :return:
        """
        df = pd.DataFrame()
        self._df['log_ram'] = np.log(self._df.ram)


        g = sns.pairplot(self._df, diag_kind='hist', kind='scatter', vars=['battery_power','log_ram','wifi','price'])
        plt.gcf().suptitle("Features relation")
        plt.show()

    def show_4_dimensions_plot(self):
        """
        Show 4 dimension plot where column x is device width, y is device height,
        the size of the circles is the cores and the color is based on the log(price)
        :return:
        """
        self._df['price_log'] = np.log(self._df['price'])
        plt.scatter(x=self._df['px_width'], y=self._df['px_height'], s=(self._df['cores_ord'] + 1) * 100,
                    c=self._df['price_log'], alpha=0.3,
                    cmap='RdGy')

        plt.title("Width vs Height")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.colorbar(label='log$_{10}$(Price)')
        for i, val in enumerate(['single','dual','triple','quad','penta','hexa','hepta','octa']):
            plt.scatter([], [], c='k', alpha=0.3, s=(i+1)*10,
                        label=str(val))
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Cores')
        plt.show()

    def price_to_price_2(self, value):
        """
        Function used to pass from price to price_2
        :param value: camera calue
        :return: transformation rate
        """
        if value <= 15:
            return 0.95
        return 1.1

    def check_transformation(self):
        """
        Calculated price rate, price_transformation and is_diff to display relation
        between camera and price_2. Displays error rate for new calculation

        Method used to analyze transformation of price_2
        :return:
        """
        self._df = self._get_merged_df()

        self._df['price_rate'] = np.round(self._df['price_2'] / self._df['price'], 2)

        # Our price transformation
        self._df['price_transformation'] = np.round(self._df.price * self._df.camera.apply(self.price_to_price_2), 2)

        # Boolean that reflects if our transformation is the same as used to calculate price_2
        self._df['is_diff'] = np.abs(self._df['price_2'] - self._df['price_transformation']) > 10

        corr_list = self._get_ordered_correlative_by_key(key='is_diff')
        corr_list.to_csv('output_files/task_4_correlative_to_price_rate.csv')
        self._show_error_rate()
        self._show_camera_above_15_rate()
        self.show_relation_by_keys(x='camera', y='price_rate', title='Camera vs Price rate')

    def _get_merged_df(self):
        transformed_df = pd.read_csv("mobile_price_2.csv", sep=",", index_col="id")
        return pd.merge(self._df, transformed_df, on='id', how='left')

    def _show_error_rate(self):
        """
        Outputs the error rate on price_2 calculation
        :return:
        """
        np.round(self._df['is_diff'].value_counts(normalize=True) * 100, 1).to_csv('output_files/task_4_error_rate_price_2_calc.csv')

    def _show_camera_above_15_rate(self):
        self._df['camera_above_15'] = self._df['camera'] > 15
        np.round(self._df['camera_above_15'].value_counts(normalize=True) * 100, 1).to_csv('output_files/task_4_camera_above_15_quantity.csv')


if __name__ == '__main__':
    mbp = Task_4()
    mbp.show_4_features_relation()
    # mbp.show_4_dimensions_plot()
    # mbp.check_transformation()
