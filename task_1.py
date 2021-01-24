import numpy as np
from mobile_prices_handler import MobilePricesHandler
import matplotlib.pyplot as plt
import seaborn as sns


class Task_1(MobilePricesHandler):

    def add_columns(self):
        """
        Add columns:
        * Resolution: Total resolution of each device
        * DPI_w: holds the DPI (dots per inch) of the screen width
        * call_ratio: ratio battery_power/talk_time
        * memory: hold the memory in GB instead of MB
        """
        self._df['resolution'] = self._df['px_width'] * self._df['px_height']
        self._df['DPI_w'] = np.round(2.54 * self._df['px_width'] /self._df['sc_w'], 1)
        self._df['call_ratio'] = self._df['battery_power'] / self._df['talk_time']
        self._df['memory'] /= 1024

    def show_describe(self):
        """
        Displays describe of dataframe
        """
        describe = self._df.describe()
        describe.to_csv('output_files/task_1.7_describe.csv')
        print(describe)

    def show_histogram(self):
        """
        Display history plot of dataframe
        """
        sns.histplot(self._df.price, bins=20, kde=True).set_title('Price histogram')
        plt.show()



if __name__ == '__main__':
    mbp = Task_1()
    #3.	Add a column that holds the total screen resolution for each device. Name it resolution.
    #4.	Add a column that holds the DPI (dots per inch) of the screen width and name it DPI_w.
    #5.	Add a column that holds the ratio battery_power/talk_time and name it call_ratio.
    #6.	Change the memory column to hold the memory in GB instead of MB.
    mbp.add_columns()
    #7.	Include the output of the `describe()` function of the dataframe.
    mbp.show_describe()
    #8.	Include a histogram of the prices.
    mbp.show_histogram()
