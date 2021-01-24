from mobile_prices_handler import MobilePricesHandler

class Task_3(MobilePricesHandler):

    def output_csv(self):
        """
        Outputs the current dataframe to csv
        :return:
        """
        self._df.to_csv("mobile_prices_converted.csv")

if __name__ == '__main__':
    mbp = Task_3()
    # 1. For each ordinal feature <O>, add a column to the
    # dataframe which holds the ordered values representing
    # each original value of F. This new column will be named
    # <O>_ord. (without the triangle brackets)
    #2. For each nominal feature <N>, add a binary column
    # OR one-hot encoding (whichever is relevant for that feature)
    # to the dataframe representing the original values.
    # Name binary columns <N>_bin, and prefix one-hot
    # encodings with <N>. (without the triangle brackets)
    mbp.map_categorical_columns()
    #3. Plot a correlation heatmap of the modified data set and include it.
    mbp.show_correlation_heatmap(title="Correlation heatmap with modified data")
    #4. Save the entire dataframe to a csv file named “mobile_prices_converted.csv” and include it in the submission.
    # Make sure you don’t add a redundant index column.
    mbp.output_csv()

