from pandas import np
from ex_3.mobile_prices_handler import MobilePricesHandler
import matplotlib.pyplot as plt


class Task_1(MobilePricesHandler):

    def add_columns(self):
        # 3.	Add a column that holds the total screen resolution for each
        # device. Name it resolution.
        self._df['resolution'] = self._df['px_width'] * self._df['px_height']
        # 4.	Add a column that holds the DPI (dots per inch)
        # of the screen width and name it DPI_w.
        self._df['DPI_w'] = np.round(2.54 * self._df['px_width'] /self._df['sc_w'], 1)
        # 5.	Add a column that holds the ratio
        # battery_power/talk_time and name it call_ratio.
        self._df['call_ratio'] = self._df['battery_power'] / self._df['talk_time']
        # 6.	Change the memory column to hold
        # the memory in GB instead of MB.
        self._df['memory'] /= 1024

    def show_describe(self):
        #7.	Include the output of the `describe()`
        # function of the dataframe.
        describe = self._df.describe()
        describe.to_csv('task_1.7_describe.csv')
        print(describe)

    def show_histogram(self):
        #8.	Include a histogramof the prices.
        self._df.hist(column='price')
        plt.show()



if __name__ == '__main__':
    mbp = Task_1()
    mbp.add_columns()
    mbp.show_describe()
    mbp.show_histogram()
