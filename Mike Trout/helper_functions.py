from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Loads data into the training and test sets necessary
def non_modular_date(date):
  '''
  inputs: the string from retrosheet.org with the date
  outputs: a number that represents the (approximate) number of days since 2000
  '''
  month = int(date[:2])
  day = int(date[3:5])
  year = int(date[6:10])

  return day + 30.5 * (month + 12 * year)

def make_dataframe(f_name):
  df = pd.read_csv(f_name, dtype=str)  # make a dataframe of Mike Trout's data
  df = df.dropna()  # remove all rows with nan's
  df['Date'] = df['Date'].apply(non_modular_date);  # convert the dates from modular to regular

  # Converts all values to numerics
  for col_name in df.columns.values:
    df[col_name] = pd.to_numeric(df[col_name])
  return df


def make_train_test(df, testing_proportion):
  '''
  Returns an 1D list of the x training, y training, x testing, and y
  y testing data for the given dataframe. THIS IS HYPERSPECIFIC TO
  THIS PROJECT, AND SHOULD NOT BE EXTENDED TO OTHER INSTANCES WITHOUT
  SIGNIFICANT CAUTION!!!
  '''
  #### NOT MY CODE ####
  training_data_df, test_data_df = train_test_split(df, test_size=testing_proportion)

  # Pull out columns for X (data to train with) and Y (value to predict)
  X_training = training_data_df.drop('Draftkings Score', axis=1).values
  Y_training = training_data_df[['Draftkings Score']].values

  # Pull out columns for X (data to train with) and Y (value to predict)
  X_testing = test_data_df.drop('Draftkings Score', axis=1).values
  Y_testing = test_data_df[['Draftkings Score']].values

  # All data needs to be scaled to a small range like 0 to 1 for the neural
  # network to work well. Create scalers for the inputs and outputs.
  X_scaler = MinMaxScaler(feature_range=(0, 1))
  Y_scaler = MinMaxScaler(feature_range=(0, 1))

  # Scale both the training inputs and outputs
  X_scaled_training = X_scaler.fit_transform(X_training)
  Y_scaled_training = Y_scaler.fit_transform(Y_training)

  # It's very important that the training and test data are scaled with the same scaler.
  X_scaled_testing = X_scaler.transform(X_testing)
  Y_scaled_testing = Y_scaler.transform(Y_testing)

  return [X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing]
