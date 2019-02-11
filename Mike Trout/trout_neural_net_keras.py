import keras
import random
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from csv import writer

######################### Creating the Neural Net #########################
def make_neural_net(file_name, learn_rate, training_epochs, layers):
  '''
  inputs: the name of the csv file with the player's data
  outputs: a neural net object that is trained with the 80% of the data
  how it works: basically follow regular protocol for making a neural net
                using tensorflow. This method is certainly not set in stone,
                and may be modified for different types of players or for
                different years or whatever.
  runtime: long :))
  '''

  assert len(layers) == 9

  TESTING_PROPORTION = .1  # the proportion of the data used for training

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

  df = make_dataframe(file_name)

  #### NOT MY CODE ####
  training_data_df, test_data_df = train_test_split(df, test_size=TESTING_PROPORTION)

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

  # Defines number of inputs and number of nodes in all of the layers
  number_of_inputs, layer_1_nodes, layer_2_nodes, layer_3_nodes, \
  layer_4_nodes, layer_5_nodes, layer_6_nodes, layer_7_nodes, number_of_outputs = layers


  # Constructing the neural network's layers/defining the model
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(layers[1], input_dim=layers[0], activation='relu'))
  for i in range(2, len(layers)-1):
    model.add(keras.layers.Dense(layers[i], activation='relu'))
    model.add(keras.layers.Dropout(.3))
  model.add(keras.layers.Dense(layers[-1], activation='linear'))
  model.compile(optimizer='adam', loss='mse') # because trying to predict score, gives a loss function of MSE

  # Training the model
  model.fit(X_scaled_training,
            Y_scaled_training,
            epochs=training_epochs,
            shuffle=True,
            verbose=0)

  std_error_rate = 0.03555387959152568
  train_error_rate = model.evaluate(X_scaled_training, Y_scaled_training, verbose=0)
  test_error_rate = model.evaluate(X_scaled_testing, Y_scaled_testing, verbose=0)
  print('\n\nThe train MSE is {}.\n'.format(train_error_rate))
  print('The test MSE is {}.\n'.format(test_error_rate))
  print('Succeeded by {} on training and {} on testing\n\n'.format(std_error_rate-train_error_rate, std_error_rate-test_error_rate))

  with open('Log_keras.csv', 'a') as csv_file:
    writer(csv_file).writerow(
      [layers, learn_rate, training_epochs, train_error_rate, test_error_rate, std_error_rate])


if __name__ == '__main__':
  layers = [random.randint(20, 100) for j in range(7)]  # random number of nodes in each layer
  # layers = [48, 52, 34, 75, 83, 56, 20]
  # num_epochs = random.randint(35, 75)
  num_epochs = 80
  for i in range(10): make_neural_net('Mike Trout.csv', .001, num_epochs, [31] + layers + [1])
