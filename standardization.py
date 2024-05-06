import pandas as pd

class Preprocessor():
    def __init__(self, input_data):
        self.input_data = input_data

    def add_pct_change(self):
        # Calculate daily percentage change in closing price
        self.input_data['PCT_CHANGE'] = pd.Series(self.input_data['CLOSE']).pct_change()
        # Fill NaN values (first row of pct_change) with zero
        self.input_data = self.input_data.fillna(0)

    def standardization(self):
        # Standardize volume data due to its large scale
        self.input_data['VOLUME'] = (self.input_data['VOLUME'] - self.input_data['VOLUME'].min()) / (self.input_data['VOLUME'].max() - self.input_data['VOLUME'].min())

    def get_preprocessed_data(self):
        # Apply preprocessing methods to the data
        self.add_pct_change()
        self.standardization()
        # Return preprocessed data
        return self.input_data
